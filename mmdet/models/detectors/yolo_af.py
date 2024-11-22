from mmdet.models.detectors import SingleStageDetector
from mmdet.models import DETECTORS


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
