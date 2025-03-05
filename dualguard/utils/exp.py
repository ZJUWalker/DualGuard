from typing import List

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dualguard.utils.configs import DatasetArgs

_dataset_name_map = {}
_dataset_dra_train_label_map = {}
_dataset_dra_test_label_map = {}


class AverageMeter(object): # 
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def register_dataset(name, dra_train_label='validation', dra_test_label='test'):
    from dualguard.utils.datasets.base import FedDataset
    
    def wrapper(cls):
        assert issubclass(
            cls, FedDataset
        ), "All dataset must inherit FedDataset"
        _dataset_name_map[name] = cls
        _dataset_dra_train_label_map[name] = dra_train_label
        _dataset_dra_test_label_map[name] = dra_test_label
        return cls

    return wrapper

def get_dataset(dataset_name, tokenizer, client_ids=None, shrink_frac=1.0, completion_only=False):
    if client_ids is None:
        client_ids = []
    if ',' in dataset_name:
        dataset_names = dataset_name.split(',')
        dataset_classes = [get_dataset_class(dn) for dn in dataset_names]
        from dualguard.utils.datasets.base import MixtureFedDataset
        return MixtureFedDataset(tokenizer, client_ids, shrink_frac, dataset_names, dataset_classes)
    else:
        return get_dataset_class(dataset_name)(tokenizer=tokenizer, client_ids=client_ids, shrink_frac=shrink_frac,
                                               completion_only=completion_only)

def load_datasets(ds_args:DatasetArgs,tokenizer:AutoTokenizer)->List[DataLoader]:
    USL_dataset = get_dataset(ds_args.dataset_name,tokenizer=tokenizer, client_ids=[])
    data_loaders={}
    has_val=True
    if ds_args.dataset_name in ['codealpaca','gsm8k']:
        has_val=False
    for split in ds_args.splits:
        if not has_val and split=='validation':
            continue
        data_loader=USL_dataset.get_dataloader_unsliced(batch_size=ds_args.batch_size,max_seq_len=ds_args.max_seq_len,type=split,shuffle=ds_args.shuffle)
        data_loaders[split]=data_loader
    return data_loaders


def get_dataset_class(dataset_name):
    from dualguard.utils.datasets import datasets
    clz = datasets.FedDataset
    if dataset_name not in _dataset_name_map:
        raise AttributeError
    clz = _dataset_name_map[dataset_name]
    return clz

def get_dra_train_label(name):
    return _dataset_dra_train_label_map[name]


def get_dra_test_label(name):
    return _dataset_dra_train_label_map[name]

