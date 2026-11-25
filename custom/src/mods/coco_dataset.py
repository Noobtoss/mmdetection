from mmdet.datasets import CocoDataset as _CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module(force=True)
class CocoDataset(_CocoDataset):
    def parse_data_info(self, raw_data_info):
        categories = self.metainfo.get('categories')
        if categories is not None:
            self.cat_ids = [category['id'] for category in categories]
            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        return super().parse_data_info(raw_data_info)
