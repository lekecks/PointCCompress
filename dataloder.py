import os
import numpy as np
import h5py
import sys, glob, time
import torch
import torch.utils.data
import MinkowskiEngine as ME

from tqdm import tqdm
from torch.utils.data.sampler import Sampler


def read_h5_geo(filedir):
    pc = h5py.File(filedir, 'r')['data'][:]
    coords = pc[:, 0:3].astype('int')

    return coords

def write_h5_geo(filedir, coords):
    data = coords.astype('uint8')
    with h5py.File(filedir, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)

    return

def read_ply_ascii_geo(filedir):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError:
            continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:, 0:3].astype('int')

    return coords

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm ' + filedir)
    f = open(filedir, 'a+')
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(coords.shape[0]) + '\n')
    f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ', str(p[2]), '\n'])
    f.close()

    return


###########################################################################################################

import torch
import MinkowskiEngine as ME


def array2vector(array, step):

    array, step = array.long().cpu(), step.long().cpu()
    vector = sum([array[:, i] * (step ** i) for i in range(array.shape[-1])])

    return vector


def isin(data, ground_truth):

    device = data.device
    data, ground_truth = data.cpu(), ground_truth.cpu()
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())

    return torch.Tensor(mask).bool().to(device)


def istopk(data, nums, rho=1.0):

    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N * rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)  # must CPU.
        mask[row_indices[indices]] = True

    return mask.bool().to(data.device)


def sort_spare_tensor(sparse_tensor):

    indices_sort = np.argsort(array2vector(sparse_tensor.C.cpu(),
                                           sparse_tensor.C.cpu().max() + 1))
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort],
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0],
                                         device=sparse_tensor.device)

    return sparse_tensor_sort


def load_sparse_tensor(filedir, device):
    coords = torch.tensor(read_ply_ascii_geo(filedir)).int()
    feats = torch.ones((len(coords), 1)).float()

    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)

    return x


def scale_sparse_tensor(x, factor):
    coords = (x.C[:, 1:] * factor).round().int()
    feats = torch.ones((len(coords), 1)).float()
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)

    return x

###########################################################################################################


class InfSampler(Sampler):
   

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, feats = list(zip(*list_data))
    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

    return coords_batch, feats_batch


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):
        filedir = self.files[idx]

        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            if filedir.endswith('.h5'): coords = read_h5_geo(filedir)
            if filedir.endswith('.ply'): coords = read_ply_ascii_geo(filedir)
            feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype('int')
            # cache
            self.cache[idx] = (coords, feats)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent
        feats = feats.astype("float32")

        return (coords, feats)


def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False,
                     collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader


if __name__ == "__main__":
    
    filedirs = sorted(glob.glob(
        '/home/ubuntu/HardDisk1/point_cloud_testing_datasets/8i_voxeilzaed_full_bodies/8i/longdress/Ply/' + '*.ply'))
    test_dataset = PCDataset(filedirs[:10])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=2, shuffle=True, num_workers=1, repeat=False,
                                       collate_fn=collate_pointcloud_fn)
    for idx, (coords, feats) in enumerate(tqdm(test_dataloader)):
        print("=" * 20, "check dataset", "=" * 20,
              "\ncoords:\n", coords, "\nfeat:\n", feats)

    test_iter = iter(test_dataloader)
    print(test_iter)
    for i in tqdm(range(10)):
        coords, feats = test_iter.next()
        print("=" * 20, "check dataset", "=" * 20,
              "\ncoords:\n", coords, "\nfeat:\n", feats)