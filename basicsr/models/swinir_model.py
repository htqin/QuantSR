import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SwinIRModel(SRModel):

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
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        
        return output

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
