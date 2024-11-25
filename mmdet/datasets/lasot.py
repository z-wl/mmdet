from __future__ import absolute_import, print_function

import os
import glob
import json
import os.path as osp

import cv2
import numpy as np
import six
import torch
from numpy.array_api import float32
import re

from mmdet.datasets.pipelines import Compose, DefaultFormatBundle, Collect
from mmcv.parallel.data_container import DataContainer
from .custom import CustomDataset

from .builder import DATASETS

@DATASETS.register_module()
class LaSOT(CustomDataset):
    CLASS2ID = {
        "lasot_sub": {
            "airplane": 0,
            "basketball": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cup": 8,
            "drone": 9,
        },
        "lasot_sub2": {
            "airplane": 0,
            "basketball": 1,
            "bird": 2,
            "bottle": 3,
            "cup": 4,
            "drone": 5,
        }
    }

    def __init__(self,
                 root_dir,
                 pipeline,
                 return_meta=False,
                 split_file='lasot_sub2.json',
                 test_mode=False,
                 subset='val',
                 batch_size=2,
                 gt_norm=False,
                 gt_format='coco',
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 width=360,
                 height=480):
        # super(LaSOT, self).__init__(None, pipeline)
        assert subset in ['train', 'test', 'val'], 'Unknown subset.'

        self.root_dir = root_dir
        self.gt_norm = gt_norm
        self.gt_format = gt_format
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.ann_file = None
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        # self.subset = subset
        self.return_meta = return_meta
        self._check_integrity(root_dir)
        self.split_file = split_file.split(".")[0]
        self.pipeline = Compose(pipeline)

        self.width = width
        self.height = height
        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/*/groundtruth.txt')))
        self.seq_dirs = [os.path.join(
            os.path.dirname(f), 'img') for f in self.anno_files]
        self.seq_names = [os.path.basename(
            os.path.dirname(f)) for f in self.anno_files]

        # self.pipeline = Compose(pipeline)
        # load subset sequence names
        split_file = os.path.join(
            os.path.dirname(__file__), split_file)
        with open(split_file, 'r') as f:
            splits = json.load(f)

        self.seq_names = splits[subset]
        # self.test_seq_names = splits['test']
        # self.val_seq_names = splits['val']

        # image and annotation paths
        self.seq_dirs = [os.path.join(
            root_dir, n[:n.rfind('-')], n, 'img')
            for n in self.seq_names]
        # self.test_seq_dirs = [os.path.join(
        #     root_dir, n[:n.rfind('-')], n, 'img')
        #     for n in self.test_seq_names]
        # self.val_seq_dirs = [os.path.join(
        #     root_dir, n[:n.rfind('-')], n, 'img')
        #     for n in self.val_seq_names]

        self.anno_files = [os.path.join(
            os.path.dirname(d), 'groundtruth.txt')
            for d in self.seq_dirs]


        self.batches_num = 0
        self.video_frames = [-1]
        self.video_frames_tail = []
        self.video_batch = [-1]

        self.compute_batch_num()
        self.shape_cache = {}
        # if self.root_dir is not None:
        #     if not osp.isabs(self.ann_file):
        #         self.ann_file = osp.join(self.root_dir, self.ann_file)
        #     if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
        #         self.img_prefix = osp.join(self.root_dir, self.img_prefix)
        #     if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
        #         self.seg_prefix = osp.join(self.root_dir, self.seg_prefix)
        #     if not (self.proposal_file is None
        #             or osp.isabs(self.proposal_file)):
        #         self.proposal_file = osp.join(self.root_dir, self.proposal_file)
        if not self.test_mode:
            self._set_group_flag()

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            video_index, _, _ = self.get_inner_index(i)
            img_shape = self.get_img_shape(i, video_index)
            # print('here', img_shape)
            if img_shape[1] / img_shape[0] > 1:
                self.flag[i] = 1

    def compute_batch_num(self):
        print(self.seq_dirs)
        count_ = 0
        for path in self.seq_dirs:
            count_ += 1
            imgs = os.listdir(path)
            imgs = [img for img in imgs if img.endswith('.jpg')]
            # print(re.split('/|\\'))
            cls = LaSOT.CLASS2ID[self.split_file][re.split(r'/|\\', path)[-3]]
            imgs_num = len(imgs)
            has_tail = imgs_num % self.batch_size != 0
            batch_num = int(imgs_num / self.batch_size) + 1 if has_tail else int(imgs_num / self.batch_size)
            # print(batch_num)
            self.video_frames.append(imgs_num + self.video_frames[-1])
            self.video_frames_tail.append(imgs_num % self.batch_size)
            self.video_batch.append(batch_num + self.video_batch[-1])
            self.batches_num += batch_num

        print(self.video_frames)
        print(self.video_batch)
        print(self.batches_num)
        print(self.video_frames_tail)

    def _filter_imgs(self, min_size=32):
        pass

    def load_annotations(self, ann_file):
        pass

    def prepare_train_img(self, index):
        return self.prepare_img(index)

    def prepare_test_img(self, index):
        return self.prepare_img(index)

    def get_inner_index(self, index):
        # video_batch = getattr(self, f'{split}_video_batch')
        # video_frames_tail = getattr(self, f'{split}_video_frames_tail')
        video_index = -1
        for batch_num in self.video_batch:
            if index > batch_num:
                video_index += 1
            else:
                break

        inner_index = ((index - self.video_batch[video_index]) - 1) * self.batch_size

        inner_batch_size = self.batch_size if index != self.video_batch[video_index + 1] or self.video_frames_tail[
            video_index] == 0 \
            else self.video_frames_tail[video_index]

        return video_index, inner_index, inner_batch_size

    def get_img_shape(self, index, video_index):
        if index not in self.shape_cache:
            sample_path = os.path.join(self.seq_dirs[video_index], '00000001.jpg')
            self.shape_cache[index] = cv2.imread(sample_path).shape

        return self.shape_cache[index]

    def get_ann_info(self, index):
        video_index, inner_index, inner_batch_size = self.get_inner_index(index)
        # anno_files = getattr(self, f'{split}_anno_files')
        anno = np.loadtxt(self.anno_files[video_index], delimiter=',')
        anno_tensor = np.stack(anno[inner_index:inner_index + inner_batch_size])
        # print(anno_tensor.shape, index, self.video_batch[video_index + 1], self.video_frames_tail[video_index], inner_index, inner_batch_size, anno.shape)
        cls = LaSOT.CLASS2ID[self.split_file][re.split(r'/|\\', self.seq_dirs[video_index])[-3]]
        # labels = np.full((inner_batch_size,), cls, dtype=float32)
        # 正则化box，添加class标签。(cls, x, y, w, h)
        img_shape = self.get_img_shape(index, video_index)
        # anno_norm = np.zeros((inner_batch_size, 6))
        bbox_ = np.zeros((inner_batch_size, 4))
        if self.gt_format == 'coco':
            bbox_[:, 0] = anno_tensor[:, 0]
            bbox_[:, 1] = anno_tensor[:, 1]
            bbox_[:, 2] = anno_tensor[:, 2] + anno_tensor[:, 0]
            bbox_[:, 3] = anno_tensor[:, 3] + anno_tensor[:, 1]
        elif self.gt_format == 'yolo':
            bbox_[:, 0] = (anno_tensor[:, 0] + anno_tensor[:, 2] / 2)
            bbox_[:, 1] = (anno_tensor[:, 1] + anno_tensor[:, 3] / 2)
            bbox_[:, 2] = anno_tensor[:, 2]
            bbox_[:, 3] = anno_tensor[:, 3]
        if self.gt_norm:
            bbox_[:, 0] /= img_shape[1]
            bbox_[:, 1] /= img_shape[0]
            bbox_[:, 2] /= img_shape[1]
            bbox_[:, 3] /= img_shape[0]

        if self.test_mode:
            return {'bboxes': np.stack(bbox_), 'labels': np.full((len(bbox_),), cls, dtype=np.int64)}
        return [{'bboxes': np.expand_dims(bbox, axis=0), 'labels': np.full((1,), cls, dtype=np.int64)} for bbox in bbox_]

    def __len__(self):
        return self.batches_num

    def prepare_train_img(self, index):
        assert not self.test_mode
        return self.prepare_img(index)

    def prepare_test_img(self, index):
        assert self.test_mode
        return self.prepare_img(index)

    def prepare_img(self, index):
        # seq_names = getattr(self, f'{split}_seq_names')
        #
        # seq_dirs = getattr(self, f'{split}_seq_dirs')

        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        video_index, inner_index, inner_batch_size = self.get_inner_index(index)

        img_files = sorted(glob.glob(os.path.join(self.seq_dirs[video_index], '*.jpg')))

        img_files = img_files[inner_index:inner_index + inner_batch_size]
        imgs = []
        for img_path in img_files:
            # imgs.append(cv2.resize(cv2.imread(img_path), (self.height, self.width)))
            imgs.append(cv2.imread(img_path))

        anno = self.get_ann_info(index) if not self.test_mode else None
        # print('anno:', anno)
        img_shape = self.get_img_shape(index, video_index)
        img_meta_list = []
        gt_bboxes_list = []
        gt_labels_list = []
        img_list = []
        def pack_res(name, img_, anno_):
            results = dict(
                filename=name,
                ori_shape=img_shape,
                img=img_,
                gt_bboxes=anno_['bboxes'] if not self.test_mode else None,
                gt_labels=anno_['labels'] if not self.test_mode else None,
                anno_info=anno_ if not self.test_mode else None)

            self.pre_pipeline(results)
            results = self.pipeline(results)
            # print('res1', results)
            img_meta_list.append(results['img_metas'])
            img_list.append(results['img'])
            if not self.test_mode:
                gt_bboxes_list.append(results['gt_bboxes'])
                gt_labels_list.append(results['gt_labels'])

        # for name, img_, anno_ in zip(img_files, imgs, anno):
        #     pack_res(name, img_, anno_)
        # results = dict(img_metas=DataContainer([meta.data for meta in img_meta_list], cpu_only=True, yolo_af=True),
        #                gt_bboxes=gt_bboxes_list, gt_labels=gt_labels_list, img=np.stack(img_list))
        # bd = DefaultFormatBundle()
        # results = bd(results, yolo_af=True)
        # return dict(img_metas=results['img_metas'], gt_bboxes=results['gt_bboxes'], gt_labels=results['gt_labels'],
        #      img=results['img'])

        if not self.test_mode:
            for name, img_, anno_ in zip(img_files, imgs, anno):
                pack_res(name, img_, anno_)
            results = dict(img_metas=DataContainer([meta.data for meta in img_meta_list], cpu_only=True, yolo_af=True),
                           gt_bboxes=gt_bboxes_list, gt_labels=gt_labels_list, img=np.stack(img_list))
        else:
            for name, img_ in zip(img_files, imgs):
                pack_res(name, img_, None)
            results = dict(img_metas=[meta for meta in img_meta_list],
                           gt_bboxes=gt_bboxes_list, gt_labels=gt_labels_list, img=torch.stack([img_l[0] for img_l in img_list]))

        if not self.test_mode:
            bd = DefaultFormatBundle()
            results = bd(results, yolo_af=True)
            return dict(img_metas=results['img_metas'], gt_bboxes=results['gt_bboxes'], gt_labels=results['gt_labels'],
                 img=results['img'])
        else:
            return dict(img_metas=results['img_metas'], img=[results['img']])


        # if self.return_meta:
        #     meta = self._fetch_meta(seq_dirs[index])
        #     return torch.from_numpy(imgs).permute(0, 3, 1, 2), anno, meta
        # else:
        #     return torch.from_numpy(imgs).permute(0, 3, 1, 2), anno

    def _check_integrity(self, root_dir):
        seq_names = os.listdir(root_dir)
        seq_names = [n for n in seq_names if not n[0] == '.']

        if os.path.isdir(root_dir) and len(seq_names) > 0:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        seq_dir = os.path.dirname(seq_dir)
        meta = {}

        # attributes
        for att in ['full_occlusion', 'out_of_view']:
            att_file = os.path.join(seq_dir, att + '.txt')
            meta[att] = np.loadtxt(att_file, delimiter=',')

        # nlp
        nlp_file = os.path.join(seq_dir, 'nlp.txt')
        with open(nlp_file, 'r') as f:
            meta['nlp'] = f.read().strip()

        return meta


# class LaSOT(object):
#     r"""`LaSOT <https://cis.temple.edu/lasot/>`_ Datasets.
#
#     Publication:
#         ``LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking``,
#         H. Fan, L. Lin, F. Yang, P. Chu, G. Deng, S. Yu, H. Bai,
#         Y. Xu, C. Liao, and H. Ling., CVPR 2019.
#
#     Args:
#         root_dir (string): Root directory of dataset where sequence
#             folders exist.
#         subset (string, optional): Specify ``train`` or ``test``
#             subset of LaSOT.
#     """
#     CLASS2ID = {
#         "lasot_sub": {
#             "airplane": 0,
#             "basketball": 1,
#             "bicycle": 2,
#             "bird": 3,
#             "boat": 4,
#             "bottle": 5,
#             "bus": 6,
#             "car": 7,
#             "cup": 8,
#             "drone": 9,
#         },
#         "lasot_sub2": {
#             "airplane": 0,
#             "basketball": 1,
#             "bird": 2,
#             "bottle": 3,
#             "cup": 4,
#             "drone": 5,
#         }
#     }
#
#
#     def __init__(self,
#                  root_dir,
#                  subset='test',
#                  return_meta=False,
#                  split_file='lasot.json',
#                  batch_size=32,
#                  width=360,
#                  height=480):
#         super(LaSOT, self).__init__()
#         assert subset in ['train', 'test', 'val'], 'Unknown subset.'
#
#         self.root_dir = root_dir
#         self.batch_size = batch_size
#         self.subset = subset
#         self.return_meta = return_meta
#         self._check_integrity(root_dir)
#         self.split_file = split_file.split(".")[0]
#
#         self.width = width
#         self.height = height
#         self.anno_files = sorted(glob.glob(
#             os.path.join(root_dir, '*/*/groundtruth.txt')))
#         self.seq_dirs = [os.path.join(
#             os.path.dirname(f), 'img') for f in self.anno_files]
#         self.seq_names = [os.path.basename(
#             os.path.dirname(f)) for f in self.anno_files]
#
#         # load subset sequence names
#         split_file = os.path.join(
#             os.path.dirname(__file__), split_file)
#         with open(split_file, 'r') as f:
#             splits = json.load(f)
#         self.seq_names = splits[subset]
#
#         # image and annotation paths
#         self.seq_dirs = [os.path.join(
#             root_dir, n[:n.rfind('-')], n, 'img')
#             for n in self.seq_names]
#         self.anno_files = [os.path.join(
#             os.path.dirname(d), 'groundtruth.txt')
#             for d in self.seq_dirs]
#
#         self.batches_num = 0
#         self.video_frames = [-1]
#         self.video_frames_tail = []
#         self.video_batch = [-1]
#         self.compute_batch_num()
#
#     def compute_batch_num(self):
#         print(self.seq_dirs)
#         count_ = 0
#         for path in self.seq_dirs:
#             count_ += 1
#             imgs = os.listdir(path)
#             imgs = [img for img in imgs if img.endswith('.jpg')]
#             cls = LaSOT.CLASS2ID[self.split_file][path.split('\\')[-3]]
#             imgs_num = len(imgs)
#             has_tail = imgs_num % self.batch_size != 0
#             batch_num = int(imgs_num / self.batch_size) + 1 if has_tail else int(imgs_num / self.batch_size)
#             # print(batch_num)
#             self.video_frames.append(imgs_num + self.video_frames[-1])
#             self.video_frames_tail.append(imgs_num % self.batch_size)
#             self.video_batch.append(batch_num + self.video_batch[-1])
#             self.batches_num += batch_num
#
#         print(self.video_frames)
#         print(self.video_batch)
#         print(self.batches_num)
#         print(self.video_frames_tail)
#         pass
#
#     # b650mk+7500f   ￥1599
#     # 32g 威刚ddr5    ￥735
#     # 映众rtx4070s 曜夜x3 $4799
#     # 威刚 翼龙s50pro 1TB ￥399
#     # 爱国者电竞es750w 金牌全电压 ￥379
#     #
#
#     def __getitem__(self, index):
#         r"""
#         Args:
#             index (integer or string): Index or name of a sequence.
#
#         Returns:
#             tuple: (img_files, anno) if ``return_meta`` is False, otherwise
#                 (img_files, anno, meta), where ``img_files`` is a list of
#                 file names, ``anno`` is a N x 4 (rectangles) numpy array, while
#                 ``meta`` is a dict contains meta information about the sequence.
#         """
#         if isinstance(index, six.string_types):
#             if not index in self.seq_names:
#                 raise Exception('Sequence {} not found.'.format(index))
#             index = self.seq_names.index(index)
#
#         video_index = -1
#         for batch_num in self.video_batch:
#             if index > batch_num:
#                 video_index += 1
#             else:
#                 break
#
#         # print('name:', self.seq_dirs[index].split('\\')[-3])
#         cls = LaSOT.CLASS2ID[self.split_file][self.seq_dirs[video_index].split('\\')[-3]]
#         img_files = sorted(glob.glob(os.path.join(
#             self.seq_dirs[video_index], '*.jpg')))
#         anno = np.loadtxt(self.anno_files[video_index], delimiter=',')
#         # shape:(h, w, c)
#         img_shape = cv2.imread(img_files[0]).shape
#
#         inner_index = ((index - self.video_batch[video_index]) - 1) * self.batch_size
#
#         inner_batch_size = self.batch_size if index != self.video_batch[video_index + 1] or self.video_frames_tail[video_index] == 0 \
#             else self.video_frames_tail[video_index]
#
#         anno_tensor = np.stack(anno[inner_index:inner_index + inner_batch_size])
#         # print(anno_tensor.shape, index, self.video_batch[video_index + 1], self.video_frames_tail[video_index], inner_index, inner_batch_size, anno.shape)
#
#         # 正则化box，添加class标签。(cls, x, y, w, h)
#         anno_norm = np.zeros((inner_batch_size, 6))
#         anno_norm[:, 1] = cls
#         anno_norm[:, 2] = (anno_tensor[:, 0] + anno_tensor[:, 2] / 2) / img_shape[1]
#         anno_norm[:, 3] = (anno_tensor[:, 1] + anno_tensor[:, 3] / 2) / img_shape[0]
#         anno_norm[:, 4] = anno_tensor[:, 2] / img_shape[1]
#         anno_norm[:, 5] = anno_tensor[:, 3] / img_shape[0]
#
#         img_files = img_files[inner_index:inner_index + inner_batch_size]
#         imgs = []
#         for img_path in img_files:
#             imgs.append(cv2.resize(cv2.imread(img_path), (self.height, self.width)))
#
#         # print(imgs[0].shape)
#         imgs = np.stack(imgs)
#         # print(torch.from_numpy(imgs).shape)
#         if self.return_meta:
#             meta = self._fetch_meta(self.seq_dirs[index])
#             return torch.from_numpy(imgs).permute(0, 3, 1, 2), torch.from_numpy(anno_norm), meta
#         else:
#             return torch.from_numpy(imgs).permute(0, 3, 1, 2), torch.from_numpy(anno_norm)
#
#     def __len__(self):
#         # return len(self.seq_names)
#         return self.batches_num
#
#     def _check_integrity(self, root_dir):
#         seq_names = os.listdir(root_dir)
#         seq_names = [n for n in seq_names if not n[0] == '.']
#
#         if os.path.isdir(root_dir) and len(seq_names) > 0:
#             # check each sequence folder
#             for seq_name in seq_names:
#                 seq_dir = os.path.join(root_dir, seq_name)
#                 if not os.path.isdir(seq_dir):
#                     print('Warning: sequence %s not exists.' % seq_name)
#         else:
#             # dataset not exists
#             raise Exception('Dataset not found or corrupted.')
#
#     def _fetch_meta(self, seq_dir):
#         seq_dir = os.path.dirname(seq_dir)
#         meta = {}
#
#         # attributes
#         for att in ['full_occlusion', 'out_of_view']:
#             att_file = os.path.join(seq_dir, att + '.txt')
#             meta[att] = np.loadtxt(att_file, delimiter=',')
#
#         # nlp
#         nlp_file = os.path.join(seq_dir, 'nlp.txt')
#         with open(nlp_file, 'r') as f:
#             meta['nlp'] = f.read().strip()
#
#         return meta

if __name__ == '__main__':
    LaSOT(
        root_dir='D:\Datasets\LaSOT\LaSOTBenchmark',
        return_meta=False,
        split_file='lasot_sub2.json',
        batch_size=96,
    )