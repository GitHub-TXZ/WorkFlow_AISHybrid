from clseg import ClSeg
import torch.nn as nn
from torch.nn.init import trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F

class InitWeights_He(object):

    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(
                module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)



class SegmentationNetwork(nn.Module):
    """
    All Segmentation Networks
    """
    def __init__(self):
        super().__init__()
        self.deep_supervision = False

        pool_op_kernel_sizes = [
            [1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]
        ]
        conv_kernel_size = [
            [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]
        ]
        self.network = ClSeg(
            1, 16, 2, len(pool_op_kernel_sizes[1:]), 2, 2, nn.Conv3d, nn.InstanceNorm3d,
            {'eps': 1e-5, 'affine': True}, nn.Dropout3d, {'p': 0, 'inplace': True},
            nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, True, False,
            lambda x: x, InitWeights_He(1e-2), pool_op_kernel_sizes[1:], conv_kernel_size,
            False, True, True
        ).cuda(0)

    def forward(self, x):
        seg_output = self.network(x)
        if self.deep_supervision:
            if not isinstance(seg_output, list) and not isinstance(seg_output, tuple):
                return [seg_output]
            else:
                return seg_output
        else:
            if not isinstance(seg_output, list) and not isinstance(seg_output, tuple):
                return seg_output
            else:
                return seg_output[0]


model = SegmentationNetwork().eval()
checkpoint = torch.load('clseg-net.pth')
print(checkpoint.keys())
model.load_state_dict(checkpoint['network_weights'])   
x = np.load('test_input\\0021092.npy')
result = np.zeros(x.shape)
print("*************predicting start****************")
x_ = torch.tensor(x).cuda(0)
x_ = F.interpolate(x_.unsqueeze(0), size=(16, 320, 320), mode='trilinear', align_corners=False)
out = model(x_)
out = F.interpolate(out, size=(x.shape[-3], x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=False).squeeze(0)
out = torch.argmax(out, dim=0).unsqueeze(0)
out = out.cpu().detach().numpy()
result = out.astype(np.float32)
print("*************predicting finished****************")
gt = np.load('test_input\\0021092_seg.npy')
gt[gt==-1] = 0
print(gt.shape)
dice = 2 * np.sum(gt* result) / (np.sum(gt) + np.sum(result))
print("Dice", dice)


import nibabel as nib
nii = nib.Nifti1Image(result.squeeze(0), np.eye(4))
nii2 = nib.Nifti1Image(x.squeeze(0), np.eye(4))
nib.save(nii2, './test_input/0021092.nii.gz')
nib.save(nii, './test_output/clseg_net_result.nii.gz')