import logging
import torch
from os import path as osp

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

    def set_layer_version(model=model, block_num=opt['network_g']['num_block']):
        # Specified for QuantSR-C (4-bit, 4x), determined by training
        names = ['body.31', 'body.30', 'body.29', 'body.8', 'body.27', 'body.13', 'body.28', 'body.15', 'body.7', 'body.10', 'body.18', 'body.17', 'body.19', 'body.20', 'body.22', 'body.6', 'body.11', 'body.16', 'body.12', 'body.23', 'body.24', 'body.25', 'body.21', 'body.14', 'body.9', 'body.26', 'body.5', 'body.4', 'body.3', 'body.2', 'body.1', 'body.0']
        list_version = []
        _list_version = []
        for (name, module) in model.net_g.named_modules():
            _list_version.append((name, module))
        for name in names:
            for _ in _list_version:
                if _[0] == name:
                    list_version.append((_[1].learnable_shortcut, _[0], _[1]))
                    continue
        for ver1 in range(block_num):
            if ver1 > block_num // 2:
                list_version[ver1][2].skip[1] = True
        for ver2 in range(block_num):
            if ver2 > block_num // 4:
                list_version[ver2][2].skip[2] = True

    if 'slimmable_version' in opt:
        print ("Set as slimmable")
        set_layer_version()
        model.set_net_version(version=opt['slimmable_version'])

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        self_ensemble = False
        if 'selfensemble_testing' in opt['val']:
            self_ensemble = opt['val']['selfensemble_testing']
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'], self_ensemble=self_ensemble)


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
