import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if 'fp_pretrain_network_g' in self.opt['path']:
            load_path_fp = self.opt['path']['fp_pretrain_network_g']
            load_net = torch.load(load_path_fp, map_location=lambda storage, loc: storage)
            model_dict =  self.net_g.state_dict()
            load_name = ['conv_last', 'conv_after_body', 'upsample']
            state_dict = {}
            for k, v in load_net['params'].items():
                if k in model_dict.keys() and (load_name[0] in k or load_name[1] in k or load_name[2] in k):
                    # v.requires_grad = False
                    state_dict[k] = v
            print (state_dict.keys())
            model_dict.update(state_dict)
            self.net_g.load_state_dict(model_dict, strict=False)
            
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def set_net_version(self, version=0):
        if "EDSR" in self.opt['network_g']['type']:
            if self.opt['num_gpu'] != 1:
                name_head = 'module.body.'
            else:
                name_head = 'body.'
            names = [name_head+str(i) for i in range(self.opt['network_g']['num_block'])]
        elif "CAT" in self.opt['network_g']['type']:
            if self.opt['num_gpu'] != 1:
                name_head = 'module.layers.'
            else:
                name_head = 'layers.'
            names = [name_head+str(i) for i in range(len(self.opt['network_g']['depth']))]
        elif "SwinIR" in self.opt['network_g']['type']:
            if self.opt['num_gpu'] != 1:
                name_head = 'module.layers.'
            else:
                name_head = 'layers.'
            names = [name_head+str(i) for i in range(len(self.opt['network_g']['depths']))]
        for (name, module) in self.net_g.named_modules():
            if name in names:
                module.version = version

    def optimize_parameters_slimmable(self, current_iter, alphas = [1/3, 1/3, 1/3]):
        self.optimizer_g.zero_grad()
        version = 0
        l_total_t = 0

        for alpha in alphas:
            self.set_net_version(version)
            version += 1
            self.output = self.net_g(self.lq)
            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix * alpha
                loss_dict['l_pix'] = l_pix * alpha
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_total += l_percep * alpha
                    loss_dict['l_percep'] = l_percep * alpha
                if l_style is not None:
                    l_total += l_style * alpha
                    loss_dict['l_style'] = l_style * alpha
            l_total_t += l_total
        l_total = l_total_t
        version = 0
        self.set_net_version(version)
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def optimize_parameters_kd(self, current_iter, features_in_hook, features_in_hook_teacher):
        self.optimizer_g.zero_grad()
        
        self.output = self.net_g(self.lq)

        # print (len(features_in_hook), len(features_in_hook_teacher))
        if len(features_in_hook) != len(features_in_hook_teacher):
            # print (len(features_in_hook), len(features_in_hook_teacher))
            # print (features_in_hook[0][0].shape, features_in_hook[1][0].shape, features_in_hook[2][0].shape, features_in_hook[3][0].shape,features_in_hook[4][0].shape, features_in_hook[5][0].shape, features_in_hook[6][0].shape, features_in_hook[7][0].shape,features_in_hook[8][0].shape, features_in_hook[9][0].shape, features_in_hook[10][0].shape, features_in_hook[11][0].shape,features_in_hook[12][0].shape, features_in_hook[13][0].shape, features_in_hook[14][0].shape, features_in_hook[15][0].shape)
            # print (features_in_hook_teacher[0][0].shape, features_in_hook_teacher[1][0].shape, features_in_hook_teacher[2][0].shape, features_in_hook_teacher[3][0].shape, features_in_hook_teacher[4][0].shape, features_in_hook_teacher[5][0].shape, features_in_hook_teacher[6][0].shape, features_in_hook_teacher[7][0].shape)
            # class_num = int(len(features_in_hook) / len(features_in_hook_teacher))
            # features_in_hook_new = []
            # for i in range(len(features_in_hook_teacher)):
            #     _ = features_in_hook[i][0]
            #     for j in range(class_num):
            #         print (class_num, ' ', i, " ", j)
            #         print (i+(j+1)*len(features_in_hook_teacher))
            #         print (_.shape, features_in_hook[i+(j+1)*len(features_in_hook_teacher)][0].shape)
            #         _ = torch.cat((_, features_in_hook[i+(j+1)*len(features_in_hook_teacher)][0]), 0)
            #     features_in_hook_new.append((_))
            # features_in_hook = features_in_hook_new
            print (len(features_in_hook), len(features_in_hook_teacher))
            # print (features_in_hook[0][0].shape, features_in_hook[1][0].shape)
            # print (features_in_hook_teacher[0][0].shape)

        def loss_term(A):
            A = A[0]
            a = torch.abs(A)
            Q = a * a
            return Q

        def total_loss(Q_s, Q_t):
            Q_s = loss_term(Q_s)
            Q_t = loss_term(Q_t)
            Q_s_norm = Q_s / torch.norm(Q_s, p=2)
            Q_t_norm = Q_t / torch.norm(Q_t, p=2)
            tmp = Q_s_norm - Q_t_norm
            loss = torch.norm(tmp, p=2)
            return loss

        loss_distill = []
        for index in range(len(features_in_hook)):
            loss_distill.append(total_loss(features_in_hook[index], features_in_hook_teacher[index]))
        loss_distill = sum(loss_distill) / len(loss_distill)
        # print ("loss_distill ", loss_distill)

        l_total = 0 + loss_distill * 1e-4
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # print (l_total, loss_distill)
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test_se(self):
        with torch.no_grad():
            # from https://github.com/thstkdgus35/EDSR-PyTorch
            lr_list = [self.lq]
            for tf in 'v', 'h', 't':
                lr_list.extend([self._test_transform(t, tf) for t in lr_list])

            sr_list = [self._test_pad(aug) for aug in lr_list]
            for i in range(len(sr_list)):
                if i > 3:
                    sr_list[i] = self._test_transform(sr_list[i], 't')
                if i % 4 > 1:
                    sr_list[i] = self._test_transform(sr_list[i], 'h')
                if (i % 4) % 2 == 1:
                    sr_list[i] = self._test_transform(sr_list[i], 'v')

            output_cat = torch.cat(sr_list, dim=0)
            self.output = output_cat.mean(dim=0, keepdim=True)
                

    def _test_transform(self, v, op):
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(v.device)

        return ret

    def _test_pad(self, lq):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                output = self.net_g_ema(lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                output = self.net_g(lq)
            self.net_g.train()
        
        return output

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, self_ensemble=False):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, self_ensemble=self_ensemble)

    def dist_validation_slimmable(self, dataloader, current_iter, tb_logger, save_img, self_ensemble=False):
        if self.opt['rank'] == 0:
            self.nondist_validation_slimmable(dataloader, current_iter, tb_logger, save_img, self_ensemble=self_ensemble)
    
    def dist_validation_vis(self, dataloader, current_iter, tb_logger, save_img, self_ensemble=False):
        if self.opt['rank'] == 0:
            self.nondist_validation_vis(dataloader, current_iter, tb_logger, save_img, self_ensemble=self_ensemble)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, self_ensemble=False):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self_ensemble == True:
                self.test_se()
            else:
                self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
    
    def nondist_validation_slimmable(self, dataloader, current_iter, tb_logger, save_img, self_ensemble=False):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            if not hasattr(self, 'metric_results_v1'):  # only execute in the first run
                self.metric_results_v1 = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            if not hasattr(self, 'metric_results_v2'):  # only execute in the first run
                self.metric_results_v2 = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.metric_results_v1 = {metric: 0 for metric in self.metric_results_v1}
            self.metric_results_v2 = {metric: 0 for metric in self.metric_results_v2}
        
        # version 0
        self.set_net_version(version=0)
        metric_data = dict()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self_ensemble == True:
                self.test_se()
            else:
                self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        # version 1
        self.set_net_version(version=1)
        metric_data_v1 = dict()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self_ensemble == True:
                self.test_se()
            else:
                self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data_v1['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data_v1['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results_v1[name] += calculate_metric(metric_data_v1, opt_)

        if with_metrics:
            for metric in self.metric_results_v1.keys():
                self.metric_results_v1[metric] /= (idx + 1)

            self._log_validation_metric_values_slimmable(current_iter, dataset_name, tb_logger, self.metric_results_v1)
        
        # version 2
        self.set_net_version(version=2)
        metric_data_v2 = dict()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self_ensemble == True:
                self.test_se()
            else:
                self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data_v2['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data_v2['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results_v2[name] += calculate_metric(metric_data_v2, opt_)

        if with_metrics:
            for metric in self.metric_results_v2.keys():
                self.metric_results_v2[metric] /= (idx + 1)

            self._log_validation_metric_values_slimmable(current_iter, dataset_name, tb_logger, self.metric_results_v2)

    def nondist_validation_vis(self, dataloader, current_iter, tb_logger, save_img, self_ensemble=False):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            if self_ensemble == True:
                self.test_se()
            else:
                self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
            
            break

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            # log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def _log_validation_metric_values_slimmable(self, current_iter, dataset_name, tb_logger, metric_results):
        log_str = ''
        for metric, value in metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            # log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
