import logging
import torch
from os import path as osp
import torch.nn.functional as F
import cv2
import torchvision.utils as vutils
import numpy as np

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    layer_name = 'conv'

    features_in_hook = []
    # features_out_hook = []
    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        # features_out_hook.append(fea_out)
        return None

    for (name, module) in model.net_g.named_modules():
        if layer_name in name:
            print(name)
            module.register_forward_hook(hook=hook)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        self_ensemble = False
        if 'selfensemble_testing' in opt['val']:
            self_ensemble = opt['val']['selfensemble_testing']
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'], self_ensemble=self_ensemble)
        break

    def show_feature_map(feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
        # feature_map[2].shape     out of bounds
        # feature_map = torch.sign(feature_map)
        feature_map = feature_map.detach().cpu().numpy().squeeze()  # 压缩成torch.Size([64, 55, 55])
        feature_map_num = feature_map.shape[0]  # 返回通道数
    
        for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出
            feature=feature_map[index]
            feature = np.asarray(feature* 255, dtype=np.uint8)
            feature=cv2.resize(feature, (70,70), interpolation =  cv2.INTER_NEAREST) #改变特征呢图尺寸
            feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET) #变成伪彩图
            cv2.imwrite('./input_visual/input_10_0/channel_{}.png'.format(str(index)), feature)

    def ours_weight_binarize(real_weights):
        weights = real_weights
        real_weights = real_weights.weights.view((weights.out_channels, weights.in_channels, weights.kernel_size, weights.kernel_size))
        mean = torch.mean(torch.mean(torch.mean(real_weights,dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        real_weights = real_weights - mean
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        real_weights = real_weights - binary_weights
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        return binary_weights + binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

    def sign_weight_binarize(real_weights):
        weights = real_weights
        real_weights = real_weights.weights.view((weights.out_channels, weights.in_channels, weights.kernel_size, weights.kernel_size))
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        return binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
    
    def show_weights(weights):
        weights = weights.detach().cpu().numpy().squeeze()
        weights_num = weights.shape[0]
    
        for index in range(weights_num):
            weight = weights[index]
            weight = np.asarray(weight * 255, dtype=np.uint8)
            weight = cv2.resize(weight, (192, 192), interpolation =  cv2.INTER_NEAREST)
            weight = cv2.applyColorMap(weight, cv2.COLORMAP_JET)
            cv2.imwrite('./weight_visual/fp32_weight/kernel_{}.png'.format(str(index)), weight)
    
    def show_weights_error(weights, weights_bi, weights_ours):
        weights = weights.detach().cpu().numpy().squeeze()
        weights_bi = weights_bi.detach().cpu().numpy().squeeze()
        weights_ours = weights_ours.detach().cpu().numpy().squeeze()
        weights_num = weights.shape[0]
        print (weights.shape, weights_bi.shape, weights_ours.shape)
    
        for index in range(weights_num):
            weight, weight_bi, weight_ours = weights[index], weights_bi[index], weights_ours[index]
            norm_ = np.sum(np.abs(weight))
            error_bi = np.sum(np.abs(weight - weight_bi))
            error_ours = np.sum(np.abs(weight - weight_ours))
            print (index, norm_)

    # show_feature_map(features_in_hook[10][0])
    
    # weights = model.net_g.body[1].conv1
    # # weights = ours_weight_binarize(weights)
    # weights = weights.weights.view((1, weights.out_channels, 8 * weights.kernel_size, 8 * weights.kernel_size))
    # show_weights(weights)

    weights = model.net_g.body[1].conv1
    weights_fp = weights.weights.data
    weights_bi = sign_weight_binarize(weights)
    weights_ours = ours_weight_binarize(weights)
    weights_fp = weights_fp.view((weights.out_channels, weights.in_channels, weights.kernel_size, weights.kernel_size))
    weights_bi = weights_bi.view((weights.out_channels, weights.in_channels, weights.kernel_size, weights.kernel_size))
    weights_ours = weights_ours.view((weights.out_channels, weights.in_channels, weights.kernel_size, weights.kernel_size))
    show_weights_error(weights_fp, weights_bi, weights_ours)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
