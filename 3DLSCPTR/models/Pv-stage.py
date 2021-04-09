import torch.nn as nn

from .py_utils import kp, AELoss
from config import system_configs

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 kernel_size=None, padding=None, attn_groups=None, embed_shape=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # print('-----------------')
        # print('self.stride: {}'.format(self.stride))
        # print('x.shape: {}'.format(x.shape))
        out = self.conv1(x)
        # print('conv1 out.shape: {}'.format(out.shape))
        out = self.bn1(out)
        # print('bn1 out.shape: {}'.format(out.shape))
        out = self.relu(out)
        # print('relu out.shape: {}'.format(out.shape))
        out = self.conv2(out)
        # print('conv2 out.shape: {}'.format(out.shape))
        out = self.bn2(out)
        # print('bn2 out.shape: {}'.format(out.shape))

        if self.downsample is not None:
            identity = self.downsample(x)
        # print('identity.shape:{}'.format(identity.shape))
        out += identity
        out = self.relu(out)
        return out

class model(kp):
    def __init__(self, db, flag=False, freeze=False):
        """
        res18  BasicBlock [2, 2, 2, 2]
        res34  BasicBlock [3, 4, 6, 3]
        res50  Bottleneck [3, 4, 6, 3]  invalid
        res101 Bottleneck [3, 4, 23, 3] invalid
        res152 Bottleneck [3, 8, 36, 3] invalid
        """

        layers          = system_configs.res_layers
        res_dims        = system_configs.res_dims
        res_strides     = system_configs.res_strides
        attn_dim        = system_configs.attn_dim
        dim_feedforward = system_configs.dim_feedforward

        num_queries = system_configs.num_queries  # number of joints
        drop_out    = system_configs.drop_out
        num_heads   = system_configs.num_heads
        enc_layers  = system_configs.enc_layers
        dec_layers  = system_configs.dec_layers
        kps_dim     = system_configs.kps_dim
        mlp_layers  = system_configs.mlp_layers
        fvv_cls     = 2

        aux_loss = system_configs.aux_loss
        pos_type = system_configs.pos_type
        pre_norm = system_configs.pre_norm
        return_intermediate = system_configs.return_intermediate

        if system_configs.block == 'BasicBlock':
            block = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]

        else:
            raise ValueError('invalid system_configs.block: {}'.format(system_configs.block))

        super(model, self).__init__(
            flag=flag,
            freeze=freeze,
            db=db,
            block=block,
            layers=layers,
            res_dims=res_dims,
            res_strides=res_strides,
            attn_dim=attn_dim,
            num_queries=num_queries,
            aux_loss=aux_loss,
            pos_type=pos_type,
            drop_out=drop_out,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            return_intermediate=return_intermediate,
            num_cls=fvv_cls,
            kps_dim=kps_dim,
            mlp_layers=mlp_layers
        )

class loss(AELoss):
    def __init__(self, db):
        super(loss, self).__init__(
            db=db,
            debug_path=system_configs.result_dir,
            aux_loss=system_configs.aux_loss,
            num_classes=2,
            dec_layers=system_configs.dec_layers
        )
