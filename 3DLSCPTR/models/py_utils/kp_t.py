import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .detr_loss import SetCriterion
from .matcher import build_matcher
from .misc import *
from .tools import homography_im2ipm_norm, homography_crop_resize, homography_ipmnorm2g

from sample.vis import save_debug_images_lstr3d

BN_MOMENTUM = 0.1

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class kp(nn.Module):
    def __init__(self,
                 flag=False,
                 freeze=False,
                 db=None,
                 block=None,
                 layers=None,
                 res_dims=None,
                 res_strides=None,
                 attn_dim=None,
                 num_queries=None,
                 aux_loss=None,
                 pos_type=None,
                 drop_out=0.1,
                 num_heads=None,
                 dim_feedforward=None,
                 enc_layers=None,
                 dec_layers=None,
                 pre_norm=None,
                 return_intermediate=None,
                 kps_dim=None,
                 mlp_layers=None,
                 num_cls=None,
                 norm_layer=FrozenBatchNorm2d
                 ):
        super(kp, self).__init__()
        self.flag = flag
        self.db = db
        # above all waste not used
        self.norm_layer = norm_layer

        self.inplanes = res_dims[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = AttentionConv(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        # self.conv1_embed = nn.Embedding(1, 3)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # kernel_sizes = [7, 7, 7, 7]
        kernel_sizes = [5, 5, 5, 5]
        # kernel_sizes = [3, 3, 3, 3]

        # paddings     = [3, 3, 3, 3]
        paddings     = [2, 2, 2, 2]
        # paddings     = [1, 1, 1, 1]

        attn_groups  = [1, 1, 1, 1]
        # attn_groups  = [8, 8, 8, 8]

        # B 32 90 160
        # print('res_dims[0]: {}'.format(res_dims[0]))
        self.layer1 = self._make_layer(block[0], res_dims[0], layers[0], stride=res_strides[0],
                                       kernel_size=kernel_sizes[0], padding=paddings[0], attn_groups=attn_groups[0],
                                       embed_shape=1)
        # self.layer1_ipm = self._make_layer(block[0], res_dims[0], layers[0], stride=res_strides[0],
        #                                kernel_size=kernel_sizes[0], padding=paddings[0], attn_groups=attn_groups[0],
        #                                embed_shape=1)

        # B 32 45 80
        # print('res_dims[1]: {}'.format(res_dims[1]))
        self.layer2 = self._make_layer(block[1], res_dims[1], layers[1], stride=res_strides[1],
                                       kernel_size=kernel_sizes[1], padding=paddings[1], attn_groups=attn_groups[1],
                                       embed_shape=1)
        # self.layer2_ipm = self._make_layer(block[1], res_dims[1], layers[1], stride=res_strides[1],
        #                                kernel_size=kernel_sizes[1], padding=paddings[1], attn_groups=attn_groups[1],
        #                                embed_shape=1)

        # B 64 23 40
        # print('res_dims[2]: {}'.format(res_dims[2]))
        self.layer3 = self._make_layer(block[2], res_dims[2], layers[2], stride=res_strides[2],
                                       kernel_size=kernel_sizes[2], padding=paddings[2], attn_groups=attn_groups[2],
                                       embed_shape=1)
        # self.layer3_ipm = self._make_layer(block[2], res_dims[2], layers[2], stride=res_strides[2],
        #                                kernel_size=kernel_sizes[2], padding=paddings[2], attn_groups=attn_groups[2],
        #                                embed_shape=1)


        # B 128 12 20
        # print('res_dims[3]: {}'.format(res_dims[3]))
        self.layer4 = self._make_layer(block[3], res_dims[3], layers[3], stride=res_strides[3],
                                       kernel_size=kernel_sizes[3], padding=paddings[3], attn_groups=attn_groups[3],
                                       embed_shape=1)
        # self.layer4_ipm = self._make_layer(block[3], res_dims[3], layers[3], stride=res_strides[3],
        #                                kernel_size=kernel_sizes[3], padding=paddings[3], attn_groups=attn_groups[3],
        #                                embed_shape=1)

        hidden_dim = attn_dim
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(res_dims[-1], hidden_dim, kernel_size=1)  # the same as channel of self.layer4

        self.transformer = build_transformer(hidden_dim=hidden_dim,
                                             dropout=drop_out,
                                             nheads=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             enc_layers=enc_layers,
                                             dec_layers=dec_layers,
                                             pre_norm=pre_norm,
                                             return_intermediate_dec=return_intermediate)

        self.class_embed = nn.Linear(hidden_dim, num_cls + 1)
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, kps_dim - 2, mlp_layers)
        self.height_embed = nn.Linear(hidden_dim, 1)
        self.pitch_embed  = nn.Linear(hidden_dim, 1)

        org_img_size    = np.array([self.db.org_h, self.db.org_w])
        resize_img_size = np.array([self.db.resize_h, self.db.resize_w])
        cam_pitch       = np.pi / 180 * self.db.pitch

        self.cam_height = torch.tensor(self.db.cam_height).unsqueeze_(0).expand([self.db.batch_size, 1]).type(torch.FloatTensor)
        self.cam_pitch = torch.tensor(cam_pitch).unsqueeze_(0).expand([self.db.batch_size, 1]).type(torch.FloatTensor)
        self.cam_height_default = torch.tensor(self.db.cam_height).unsqueeze_(0).expand(self.db.batch_size).type(torch.FloatTensor)
        self.cam_pitch_default = torch.tensor(cam_pitch).unsqueeze_(0).expand(self.db.batch_size).type(torch.FloatTensor)

        # image scale matrix  [0,1]x[0,1] <-> [0,479]x[0,359]
        self.S_im = torch.from_numpy(np.array([[self.db.resize_w, 0, 0],
                                               [0, self.db.resize_h, 0],
                                               [0, 0, 1]], dtype=np.float32))
        self.S_im_inv = torch.from_numpy(np.array([[1 / np.float(self.db.resize_w), 0, 0],
                                                   [0, 1 / np.float(self.db.resize_h), 0],
                                                   [0, 0, 1]], dtype=np.float32))
        self.S_im_inv_batch = self.S_im_inv.unsqueeze_(0).expand([self.db.batch_size, 3, 3]).type(torch.FloatTensor)

        # image transform matrix  [0,479]x[0,359] <-> [0,1919]x[0,1079]
        H_c      = homography_crop_resize(org_img_size, self.db.crop_y, resize_img_size)
        self.H_c = torch.from_numpy(H_c).unsqueeze_(0).expand([self.db.batch_size, 3, 3]).type(torch.FloatTensor)

        # camera intrinsic matrix
        self.K = torch.from_numpy(self.db.K).unsqueeze_(0).expand([self.db.batch_size, 3, 3]).type(torch.FloatTensor)

        # homograph ground to camera  ground(m) <-> [0,1919]x[0,1079]
        # H_g2cam = np.array([[1,                             0,               0],
        #                     [0, np.cos(np.pi / 2 + cam_pitch), args.cam_height],
        #                     [0, np.sin(np.pi / 2 + cam_pitch),               0]])
        H_g2cam = np.array([[1, 0, 0],
                            [0, np.sin(-cam_pitch), self.db.cam_height],
                            [0, np.cos(-cam_pitch), 0]])
        self.H_g2cam = torch.from_numpy(H_g2cam).unsqueeze_(0).expand([self.db.batch_size, 3, 3]).type(torch.FloatTensor)

        # transform from ipm normalized coordinates to ground coordinates  # ground(m) <-> top_view[-10,10]x[3,103]
        H_ipmnorm2g = homography_ipmnorm2g(self.db.top_view_region)
        self.H_ipmnorm2g = torch.from_numpy(H_ipmnorm2g).unsqueeze_(0).expand([self.db.batch_size, 3, 3]).type(torch.FloatTensor)

        # compute the tranformation from ipm norm coords to image norm coords
        M_ipm2im = torch.bmm(self.H_g2cam, self.H_ipmnorm2g)
        M_ipm2im = torch.bmm(self.K, M_ipm2im)
        M_ipm2im = torch.bmm(self.H_c, M_ipm2im)
        M_ipm2im = torch.bmm(self.S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(M_ipm2im, M_ipm2im[:, 2, 2].reshape([self.db.batch_size, 1, 1]).expand([self.db.batch_size, 3, 3]))
        self.M_inv = M_ipm2im

        self.M_inv              = self.M_inv.cuda()
        self.S_im               = self.S_im.cuda()
        self.S_im_inv           = self.S_im_inv.cuda()
        self.S_im_inv_batch     = self.S_im_inv_batch.cuda()
        self.H_c                = self.H_c.cuda()
        self.K                  = self.K.cuda()
        self.H_g2cam            = self.H_g2cam.cuda()
        self.H_ipmnorm2g        = self.H_ipmnorm2g.cuda()
        self.cam_height_default = self.cam_height_default.cuda()
        self.cam_pitch_default  = self.cam_pitch_default.cuda()

        size_top = torch.Size([self.db.batch_size, np.int(self.db.ipm_h), np.int(self.db.ipm_w)])
        self.project_layer = ProjectiveGridGenerator(size_top, self.M_inv)

    def _make_layer(self, block, planes, blocks, stride=1,
                    kernel_size=None, padding=None, attn_groups=None, embed_shape=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            kernel_size=kernel_size, padding=padding, attn_groups=attn_groups,
                            embed_shape=embed_shape))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                kernel_size=kernel_size, padding=padding, attn_groups=attn_groups,
                                embed_shape=embed_shape))
        return nn.Sequential(*layers)

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks  = xs[1]
        heights = xs[2]
        pitches = xs[3]
        self.update_projection(heights, pitches)
        grid = self.project_layer(self.M_inv)
        x_proj = F.grid_sample(images, grid)
        p = self.conv1(x_proj)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)
        p = self.layer3(p)
        p = self.layer4(p)

        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos    = self.position_embedding(p, pmasks)
        hs, _, weights = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)  # nheads B nqueries hdim

        output_class = self.class_embed(hs)  # nheads B nqueries num_cls
        # output_coord = self.bbox_embed(hs).sigmoid()  # nheads B nqueries num_kps
        output_coord = self.bbox_embed(hs)  # nheads B nqueries kps_dim

        latent_height = self.height_embed(hs).sigmoid() + 1.  # nheads B nqueries 1
        latent_height = torch.mean(latent_height, dim=-2, keepdim=True)  # nheads B 1 1
        latent_height = torch.mean(latent_height, dim=0, keepdim=True)  # 1, B, 1, 1

        # latent_pitch  = self.pitch_embed(hs).sigmoid() * 10 * 3.14159265359 / 180  # nheads B nqueries 1
        latent_pitch = self.pitch_embed(hs)
        latent_pitch = torch.mean(latent_pitch, dim=-2, keepdim=True)  # nheads B 1 1
        latent_pitch = torch.mean(latent_pitch, dim=0, keepdim=True)  # 1, B, 1, 1

        output_coord = torch.cat([output_coord, latent_height.repeat(output_coord.shape[0], 1, output_coord.shape[2], 1),
                                  latent_pitch.repeat(output_coord.shape[0], 1, output_coord.shape[2], 1)], dim=-1)
        # print(output_coord.shape);exit()
        # height = latent_height.squeeze().unsqueeze(dim=-1)
        # pitch  = latent_pitch.squeeze().unsqueeze(dim=-1)
        # self.update_projection(height, pitch)
        # # update transformation for data augmentation (only for training)
        # # self.update_projection_for_data_aug(aug_mats)
        # grid = self.project_layer(self.M_inv)
        # x_proj = F.grid_sample(images, grid)

        out = {'pred_logits': output_class[-1], 'pred_boxes': output_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_coord)
        return out, weights

    def _test(self, *xs, **kwargs):
        # raise NotImplementedError
        return self._train(*xs, **kwargs)

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_projection(self, cam_height, cam_pitch):
        """
            Update transformation matrix based on ground-truth cam_height and cam_pitch
            This function is "Mutually Exclusive" to the updates of M_inv from network prediction
        :param args:
        :param cam_height:
        :param cam_pitch:
        :return:
        """
        for i in range(cam_height.shape[0]):
            M, M_inv = homography_im2ipm_norm(self.db.top_view_region, np.array([self.db.org_h, self.db.org_w]),
                                              self.db.crop_y, np.array([self.db.resize_h, self.db.resize_w]),
                                              cam_pitch[i].data.cpu().numpy(), cam_height[i].data.cpu().numpy(), self.db.K)
            self.M_inv[i] = torch.from_numpy(M_inv).type(torch.FloatTensor)
        self.cam_height = cam_height
        self.cam_pitch = cam_pitch

    def update_projection_for_data_aug(self, aug_mats):
        """
            update transformation matrix when data augmentation have been applied, and the image augmentation matrix are provided
            Need to consider both the cases of 1. when using ground-truth cam_height, cam_pitch, update M_inv
                                               2. when cam_height, cam_pitch are online estimated, update H_c for later use
        """
        # if not self.no_cuda:
        aug_mats = aug_mats.cuda()

        for i in range(aug_mats.shape[0]):
            # update H_c directly
            self.H_c[i] = torch.matmul(aug_mats[i], self.H_c[i])
            # augmentation need to be applied in unnormalized image coords for M_inv
            aug_mats[i] = torch.matmul(torch.matmul(self.S_im_inv, aug_mats[i]), self.S_im)
            self.M_inv[i] = torch.matmul(aug_mats[i], self.M_inv[i])


class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda=False):
        """

        :param size_ipm: size of ipm tensor NCHW
        :param im_h: height of image tensor
        :param im_w: width of image tensor
        :param M: normalized transformation matrix between image view and IPM
        :param no_cuda:
        """
        super().__init__()
        self.N, self.H, self.W = size_ipm  # 8, 208 128
        # print(M.shape);exit()  # 8, 3, 3
        # self.im_h = im_h
        # self.im_w = im_w
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        # print(linear_points_W.shape)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)
        # print(linear_points_H.shape)
        # print(linear_points_H)
        # exit()

        # use M only to decide the type not value
        self.base_grid = M.new(self.N, self.H, self.W, 3)
        # print(self.base_grid.shape);exit()  # 8 208 128 3
        # outer_dot = torch.ger(torch.ones(self.H), linear_points_W)
        # print(outer_dot.shape)  # 208 128
        self.base_grid[:, :, :, 0] = torch.ger(
                torch.ones(self.H), linear_points_W).expand_as(self.base_grid[:, :, :, 0])
        # print(self.base_grid[:, :, :, 0])

        self.base_grid[:, :, :, 1] = torch.ger(
                linear_points_H, torch.ones(self.W)).expand_as(self.base_grid[:, :, :, 1])
        # print(self.base_grid[:, :, :, 1]);exit()


        self.base_grid[:, :, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        # if not no_cuda:
        self.base_grid = self.base_grid.cuda()
            # self.im_h = self.im_h.cuda()
            # self.im_w = self.im_w.cuda()

    def forward(self, M):
        # compute the grid mapping based on the input transformation matrix M
        # if base_grid is top-view, M should be ipm-to-img homography transformation, and vice versa
        # print(M.transpose(1,2).shape)  #  8 3 3
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), M.transpose(1, 2))
        # print(grid.shape);exit()   #8, 26624, 3
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape((self.N, self.H, self.W, 2))
        #
        """
        output grid to be used for grid_sample. 
            1. grid specifies the sampling pixel locations normalized by the input spatial dimensions.
            2. pixel locations need to be converted to the range (-1, 1)
        """
        grid = (grid - 0.5) * 2
        return grid

class AELoss(nn.Module):
    def __init__(self,
                 db,
                 debug_path=None,
                 aux_loss=None,
                 num_classes=None,
                 dec_layers=None
                 ):
        super(AELoss, self).__init__()
        self.debug_path  = debug_path
        self.db = db
        weight_dict = {'loss_ce': 1,
                       'loss_gflatpolys': 5, 'loss_gflatzsys': 5,
                       'loss_gflatlowers': 2, 'loss_gflatuppers': 2}
        # set_cost_bbox and set_cost_giou are not used, i will fix it in the next version
        # boxes key represents the polys, lowers and uppers
        # cardinality is not used to propagate loss
        matcher = build_matcher(set_cost_class=weight_dict['loss_ce'],
                                poly_weight=weight_dict['loss_gflatpolys'],
                                lower_weight=weight_dict['loss_gflatlowers'],
                                upper_weight=weight_dict['loss_gflatuppers'],
                                seq_len=self.db.max_2dpoints)
        losses  = ['labels', 'boxes', 'cardinality']
        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(num_classes=num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=1.0,
                                      losses=losses,
                                      seq_len=self.db.max_2dpoints)

    def forward(self,
                iteration,
                save,
                viz_split,
                outputs,
                targets):
        batch_size      = targets[0].shape[0]
        gt_step         = (len(targets) - 1) // batch_size
        gt_2dgflatlanes = [tgt[0] for tgt in targets[1::gt_step]]
        gt_2dgflatflags = [tgt[0] for tgt in targets[2::gt_step]]
        gt_heights      = [tgt[0] for tgt in targets[3::gt_step]]
        gt_pitches      = [tgt[0] for tgt in targets[4::gt_step]]

        loss_dict, indices = self.criterion(outputs, gt_2dgflatlanes, gt_2dgflatflags)
        weight_dict        = self.criterion.weight_dict

        losses                     = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced          = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled   = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled      = sum(loss_dict_reduced_scaled.values())
        loss_value                 = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # save = True
        if save:
            which_stack = 0
            save_dir = os.path.join(self.debug_path, viz_split)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = 'iter_{}_layer_{}'.format(iteration % 5000, which_stack)
            save_path = os.path.join(save_dir, save_name)
            with torch.no_grad():
                gt_viz_inputs = targets[0]
                # image-view
                tgt_class = [tgt[:, 0].long() for tgt in gt_2dgflatlanes]

                pred_class = outputs['pred_logits'].detach()
                prob = F.softmax(pred_class, -1)
                scores, pred_class = prob.max(-1)
                pred_boxes = outputs['pred_boxes'].detach()
                pred_polys = torch.cat([scores.unsqueeze(-1), pred_boxes], dim=-1)

                save_debug_images_lstr3d(gt_viz_inputs,
                                         tgt_boxes=gt_2dgflatlanes,
                                         tgt_class=tgt_class,
                                         pred_boxes=pred_polys,
                                         pred_class=pred_class,
                                         prefix=save_path,
                                         db=self.db,
                                         tgt_pitches=gt_pitches,
                                         tgt_heights=gt_heights,
                                         tgt_flag=gt_2dgflatflags)
            # exit()

        return (losses, loss_dict_reduced, loss_dict_reduced_unscaled,
                loss_dict_reduced_scaled, loss_value)
