"""Custom data pipelines
"""

import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomResize(object):
    """Randomly resize images & bbox & mask.

    The standard Resize operation in mmdet requires resizing to a fixed size
    This implementation resizes the images to a random ratio of the original size

    Args:
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self,
                 ratio_range=None,
                 keep_ratio=True,
                 backend='cv2'):
        self.img_scale = None
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.backend = backend

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        h, w = results['img'].shape[:2]

        min_ratio, max_ratio = self.ratio_range
        assert min_ratio <= max_ratio
        scale_factor = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        
        results['scale'] = (int(w * scale_factor), int(h * scale_factor))
        results['scale_idx'] = None

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio})'
        return repr_str
