import torch
import numpy as np
import MinkowskiEngine as ME
import importlib
import os, sys, time, logging, glob, argparse
import MinkowskiEngine as ME

from dataloder import isin, istopk, PCDataset, make_data_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import PCCModel

criterion = torch.nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.writer = SummaryWriter(log_dir=config.logdir)

        self.model = model.to(device)
        self.logger.info(model)
        self.load_state_dict()
        self.epoch = 0
        self.record_set = {'bce':[], 'bces':[], 'bpp':[],'sum_loss':[], 'metrics':[]}

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_state_dict(self):

        if self.config.init_ckpt=='':
            self.logger.info('Random initialization.')
        else:
            ckpt = torch.load(self.config.init_ckpt)
            self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)

        return

    def save_model(self):
        torch.save({'model': self.model.state_dict()},
            os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))
        return

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer

    @torch.no_grad()
    def record(self, main_tag, global_step):

        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items():
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items():
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))

        for k in self.record_set.keys():
            self.record_set[k] = []

        return

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        for _, (coords, feats) in enumerate(tqdm(dataloader)):

            x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)

            out_set = self.model(x, training=False)

            bce, bce_list = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce
                bce_list.append(curr_bce.item())
            bpp = get_bits(out_set['likelihood'])/float(x.__len__())
            sum_loss = self.config.alpha * bce + self.config.beta * bpp
            metrics = []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                metrics.append(get_metrics(out_cls, ground_truth))

            self.record_set['bce'].append(bce.item())
            self.record_set['bces'].append(bce_list)
            self.record_set['bpp'].append(bpp.item())
            self.record_set['sum_loss'].append(bce.item() + bpp.item())
            self.record_set['metrics'].append(metrics)
            torch.cuda.empty_cache()

        self.record(main_tag=main_tag, global_step=self.epoch)

        return

    def train(self, dataloader):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))

        self.optimizer = self.set_optimizer()
        self.logger.info('alpha:' + str(round(self.config.alpha,2)) + '\tbeta:' + str(round(self.config.beta,2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))

        self.logger.info('Training Files length:' + str(len(dataloader)))

        start_time = time.time()
        for batch_step, (coords, feats) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()

            x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)

            out_set = self.model(x, training=True)

            bce, bce_list = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                curr_bce = get_bce(out_cls, ground_truth)/float(out_cls.__len__())
                
                bce += curr_bce
                bce_list.append(curr_bce.item())
            bpp = get_bits(out_set['likelihood'])/float(x.__len__())
            sum_loss = self.config.alpha * bce + self.config.beta * bpp

            sum_loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                metrics = []
                for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                    metrics.append(get_metrics(out_cls, ground_truth))
                self.record_set['bce'].append(bce.item())
                self.record_set['bces'].append(bce_list)
                self.record_set['bpp'].append(bpp.item())
                self.record_set['sum_loss'].append(bce.item() + bpp.item())
                self.record_set['metrics'].append(metrics)
                if (time.time() - start_time) > self.config.check_time*60:
                    self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
                    self.save_model()
                    start_time = time.time()
            torch.cuda.empty_cache()

        with torch.no_grad(): self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
        self.save_model()
        self.epoch += 1

        return


    def get_bce(data, groud_truth):

        mask = isin(data.C, groud_truth.C)
        bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
        bce /= torch.log(torch.tensor(2.0)).to(bce.device)
        sum_bce = bce * data.shape[0]

        return sum_bce


    def get_bits(likelihood):
        bits = -torch.sum(torch.log2(likelihood))

        return bits


    def get_metrics(data, groud_truth):
        mask_real = isin(data.C, groud_truth.C)
        nums = [len(C) for C in groud_truth.decomposed_coordinates]
        mask_pred = istopk(data, nums, rho=1.0)
        metrics = get_cls_metrics(mask_pred, mask_real)

        return metrics[0]


    def get_cls_metrics(pred, real):
        TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
        FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
        FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
        TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        IoU = TP / (TP + FP + FN + 1e-7)

        return [round(precision, 4), round(recall, 4), round(IoU, 4)]


    def parse_args():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("--dataset", default='/home/ubuntu/HardDisk2/color_training_datasets/training_dataset/')
        parser.add_argument("--dataset_num", type=int, default=2e4)

        parser.add_argument("--alpha", type=float, default=1., help="weights for distoration.")
        parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

        parser.add_argument("--init_ckpt", default='')
        parser.add_argument("--lr", type=float, default=8e-4)

        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--epoch", type=int, default=50)
        parser.add_argument("--check_time", type=float, default=10, help='frequency for recording state (min).')
        parser.add_argument("--prefix", type=str, default='tp', help="prefix of checkpoints/logger, etc.")

        args = parser.parse_args()

        return args


    class TrainingConfig():
        def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, lr, check_time):
            self.logdir = logdir
            if not os.path.exists(self.logdir): os.makedirs(self.logdir)
            self.ckptdir = ckptdir
            if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
            self.init_ckpt = init_ckpt
            self.alpha = alpha
            self.beta = beta
            self.lr = lr
            self.check_time = check_time


    if __name__ == '__main__':

        args = parse_args()
        training_config = TrainingConfig(
            logdir=os.path.join('./logs', args.prefix),
            ckptdir=os.path.join('./ckpts', args.prefix),
            init_ckpt=args.init_ckpt,
            alpha=args.alpha,
            beta=args.beta,
            lr=args.lr,
            check_time=args.check_time)

        model = PCCModel()

        trainer = Trainer(config=training_config, model=model)


        filedirs = sorted(glob.glob(args.dataset + '*.h5'))[:int(args.dataset_num)]
        train_dataset = PCDataset(filedirs[round(len(filedirs) / 10):])
        train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
        test_dataset = PCDataset(filedirs[:round(len(filedirs) / 10)])
        test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)


        for epoch in range(0, args.epoch):
            if epoch > 0: trainer.config.lr = max(trainer.config.lr / 2, 1e-5)  # update lr
            trainer.train(train_dataloader)
            trainer.test(test_dataloader, 'Test')