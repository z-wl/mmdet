from mmdet.models.detectors import SingleStageDetector
from mmdet.models import DETECTORS
import torch


@DETECTORS.register_module()
class YOLOAF(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(YOLOAF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, init_cfg=init_cfg)

    def compute_mov_channel(self, x):
        # x = x.to('cuda:0')
        with torch.no_grad():
            # print(x.shape)
            img_gray_l = x[:, 0, :, :] * self.bgr_wei[0] + x[:, 1, :, :] * self.bgr_wei[1] + x[:, 2, :, :] * \
                         self.bgr_wei[2]
            # img_gray_l = torch.sum(t_img_l, dim=-1)
            img_gray_l = torch.cat([img_gray_l[0].clone().unsqueeze(0), img_gray_l])
            frames_fft = torch.fft.fftn(img_gray_l, dim=(-2, -1))
            last_frames_abs = torch.abs(frames_fft[:-1])
            this_frames_ang = torch.angle(frames_fft[1:])
            last_frames_mov = (img_gray_l[1:] - torch.abs(
                torch.fft.ifftn(torch.polar(last_frames_abs, this_frames_ang), dim=(-2, -1)))).unsqueeze(1)

        return torch.cat((x, last_frames_mov), dim=1)
