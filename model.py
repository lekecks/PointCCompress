import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
import torchac
from autoencoder import Encoder, Decoder
from torch.nn.parameter import Parameter

class RoundNoGradient(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x < 1e-9] = 0
        except RuntimeError:
            print("ERROR! grad1[x<1e-9] = 0")
            grad1 = g.clone()
        pass_through_if = np.logical_or(x.cpu().detach().numpy() >= 1e-9, g.cpu().detach().numpy() < 0.0)
        t = torch.Tensor(pass_through_if + 0.0).to(grad1.device)

        return grad1 * t


class EntropyBottleneck(nn.Module):


    def __init__(self, channels, init_scale=8, filters=(3, 3, 3)):

        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._channels = channels
        self.ASSERT = False

        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))

        self._matrices = nn.ParameterList([])
        self._biases = nn.ParameterList([])
        self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):

            self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix)
            self._matrices.append(self.matrix)
            #
            self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
            self.bias.data.copy_(init_bias)  # copy or fill?
            self._biases.append(self.bias)
            #
            self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            self.factor.data.fill_(0.0)
            self._factors.append(self.factor)

    def _logits_cumulative(self, inputs):

        logits = inputs
        for i in range(len(self._filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i])
            logits = torch.matmul(matrix, logits)
            logits += self._biases[i]
            factor = torch.tanh(self._factors[i])
            logits += factor * torch.tanh(logits)

        return logits

    def _quantize(self, inputs, mode):

        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "symbols":
            return RoundNoGradient.apply(inputs)

    def _likelihood(self, inputs):

        inputs = inputs.permute(1, 0).contiguous()  
        shape = inputs.size() 
        inputs = inputs.view(shape[0], 1, -1) 
        inputs = inputs.to(self.matrix.device)

        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(torch.add(lower, upper)).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0)

        return likelihood

    def forward(self, inputs, quantize_mode="noise"):

        if quantize_mode is None:
            outputs = inputs
        else:
            outputs = self._quantize(inputs, mode=quantize_mode)
        likelihood = self._likelihood(outputs)
        likelihood = Low_bound.apply(likelihood)

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    @torch.no_grad()
    def compress(self, inputs):

        values = self._quantize(inputs, mode="symbols")

        min_v = values.min().detach().float()
        max_v = values.max().detach().float()
        symbols = torch.arange(min_v, max_v + 1)
        symbols = symbols.reshape(-1, 1).repeat(1, values.shape[-1])  
        values_norm = values - min_v
        min_v, max_v = torch.tensor([min_v]), torch.tensor([max_v])
        values_norm = values_norm.to(torch.int16)


        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        pmf = pmf.permute(1, 0) 


        cdf = self._pmf_to_cdf(pmf)

        out_cdf = cdf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1).detach().cpu()
        strings = torchac.encode_float_cdf(out_cdf, values_norm.cpu(), check_input_bounds=True)

        return strings, min_v.cpu().numpy(), max_v.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, min_v, max_v, shape, channels):

        symbols = torch.arange(min_v, max_v + 1)
        symbols = symbols.reshape(-1, 1).repeat(1, channels)


        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)
        pmf = pmf.permute(1, 0)

        cdf = self._pmf_to_cdf(pmf)

        out_cdf = cdf.unsqueeze(0).repeat(shape[0], 1, 1).detach().cpu()
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += min_v

        return values

class PCCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F,
            coordinate_map_key=data.coordinate_map_key,
            coordinate_manager=data.coordinate_manager,
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):

        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]


        y_q, likelihood = self.get_likelihood(y,
            quantize_mode="noise" if training else "symbols")


        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {'out':out,
                'out_cls_list':out_cls_list,
                'prior':y_q,
                'likelihood':likelihood,
                'ground_truth_list':ground_truth_list}

if __name__ == '__main__':
    model = PCCModel()
    print(model)

