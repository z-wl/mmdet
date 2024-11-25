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
                 init_cfg=None,
                 use_fft=False):
        super(YOLOAF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, init_cfg=init_cfg)
        self.use_fft = use_fft
        self.last_fft_frame_abs = None
        self.bgr_wei = [0.11, 0.59, 0.3]

    def compute_mov_channel(self, x):
        # x = x.to('cuda:0')
        # if isinstance(x, list):
        #     x = x[0]
        if isinstance(x, list):
            tmp_x = x[0]
        else:
            tmp_x = x

        with torch.no_grad():
            if tmp_x.shape[0] == 1:
                # if isinstance(x, list):
                #     img_gray_l = x[0][:, 0, :, :] * self.bgr_wei[0] + x[0][:, 1, :, :] * self.bgr_wei[1] + x[0][:, 2, :, :] * \
                #              self.bgr_wei[2]
                # else:
                img_gray_l = tmp_x[:, 0, :, :] * self.bgr_wei[0] + tmp_x[:, 1, :, :] * self.bgr_wei[1] + tmp_x[:, 2, :, :] * \
                                 self.bgr_wei[2]

                frame_fft = torch.fft.fftn(img_gray_l, dim=(-2, -1))
                if self.last_fft_frame_abs is None:
                    self.last_fft_frame_abs = torch.abs(frame_fft[0])
                    # print(x.shape, img_gray_l.shape)
                    if isinstance(x, list):
                        x[0] = torch.cat((tmp_x, torch.zeros_like(img_gray_l).unsqueeze(0)), dim=1)
                    else:
                        x = torch.cat((x, torch.zeros_like(img_gray_l).unsqueeze(0)), dim=1)
                    return x
                else:
                    this_frames_ang = torch.angle(frame_fft[0])
                    # print('ang shape:', this_frames_ang.shape, img_gray_l.shape, self.last_fft_frame_abs.shape)
                    channel_mov = (img_gray_l - torch.abs(
                        torch.fft.ifftn(torch.polar(self.last_fft_frame_abs, this_frames_ang), dim=(-2, -1)))).unsqueeze(1)
                    # print('shape:', channel_mov.shape)
                    self.last_fft_frame_abs = torch.abs(frame_fft[0])
                   #  print(torch.cat((tmp_x, channel_mov), dim=1))
                    if isinstance(x, list):
                        x[0] = torch.cat((tmp_x, channel_mov), dim=1)
                    else:
                        x = torch.cat((x, channel_mov), dim=1)
                    return x
            else:

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
                # print('frames batch:', last_frames_abs.shape, this_frames_ang.shape, last_frames_mov.shape)
                return torch.cat((x, last_frames_mov), dim=1)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        if self.use_fft:
            img = self.compute_mov_channel(img)
            return super(YOLOAF, self).forward_train(img, img_metas, gt_bboxes, gt_labels)
        else:
            return super(YOLOAF, self).forward_train(img, img_metas, gt_bboxes, gt_labels)

    def forward_test(self, imgs, img_metas, **kwargs):
        if self.use_fft:
            imgs = self.compute_mov_channel(imgs)
            return super(YOLOAF, self).forward_test(imgs, img_metas, **kwargs)
        else:
            return super(YOLOAF, self).forward_test(imgs, img_metas, **kwargs)