import warnings
from .image_base import img_root_map, ImageBaseDataset
from .screen_understanding import ReferringDataset, GroundingDataset, NavigationDataset, VQA_CONFIGS
from .screen_understanding_vqa import VQADataset
from ..smp import *


class ConcatDataset(ImageBaseDataset):
    # This dataset takes multiple dataset names as input and aggregate them into a single dataset.

    DATASET_SETS = {
        'MMMB': ['MMMB_ar', 'MMMB_cn', 'MMMB_en', 'MMMB_pt', 'MMMB_ru', 'MMMB_tr'],
        'MTL_MMBench_DEV': [
            'MMBench_dev_ar', 'MMBench_dev_cn', 'MMBench_dev_en',
            'MMBench_dev_pt', 'MMBench_dev_ru', 'MMBench_dev_tr'
        ]
    }

    def __init__(self, dataset):
        datasets = self.DATASET_SETS[dataset]
        self.dataset_map = {}
        self.dataset_name = dataset
        self.datasets = datasets
        for dname in datasets:
            dataset = build_dataset(dname)
            if not dataset:
                warnings.warn(f'Build dataset failed, dataset is None')
                return
            self.dataset_map[dname] = dataset
        TYPES = [x.TYPE for x in self.dataset_map.values()]
        MODALITIES = [x.MODALITY for x in self.dataset_map.values()]
        if not np.all([x == TYPES[0] for x in TYPES]):
            warnings.warn(f'Datasets TYPES error, datasets: {datasets},  types: {TYPES}')
            return 
        
        if not np.all([x == MODALITIES[0] for x in MODALITIES]):
            warnings.warn(f'Datasets MODALITIES error, datasets: {datasets},  MODALITIES: {MODALITIES}' )
            return 
        self.TYPE = TYPES[0]
        self.MODALITY = MODALITIES[0]
        data_all = []
        for dname in datasets:
            data = self.dataset_map[dname].data
            data['SUB_DATASET'] = [dname] * len(data)
            data_new = localize_df(data, dname, nproc=16)
            data_all.append(data_new)

        data = pd.concat(data_all)
        data['original_index'] = data.pop('index')
        data['index'] = np.arange(len(data))
        self.data = data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        idx = line['original_index']
        dname = line['SUB_DATASET']
        org_data = self.dataset_map[dname].data
        org_line = cp.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].build_prompt(org_line)

    def dump_image(self, line):
        # Assert all images are pre-dumped
        if 'image' in line:
            warnings.warn(f'Error "image" in line , line : {line}')
            return 
        if 'image_path' not in line:
            warnings.warn(f'Error "image_path" not in line , line : {line}')
            return 
        tgt_path = toliststr(line['image_path'])
        return tgt_path

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SETS)

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        # First, split the eval_file by dataset
        data_all = load(eval_file)
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            data_sub = data_all[data_all['SUB_DATASET'] == dname]
            data_sub.pop('index')
            data_sub['index'] = data_sub.pop('original_index')
            data_sub.pop('SUB_DATASET')
            dump(data_sub, tgt)
        # Then, evaluate each dataset separately
        results_all = []
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            res = self.dataset_map[dname].evaluate(tgt, **judge_kwargs)
            if not  isinstance(res, pd.DataFrame):
                warnings.warn(f'Error res type {type(res)}, needed dataframe')
                return 
            res['DATASET'] = [dname] * len(res)
            results_all.append(res)
        result = pd.concat(results_all)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(result, score_file)
        return result


# Add new supported dataset class here
DATASET_CLASSES = [
    ReferringDataset, GroundingDataset, NavigationDataset, VQADataset
]


SUPPORTED_DATASETS = []
for DATASET_CLS in DATASET_CLASSES:
    SUPPORTED_DATASETS.extend(DATASET_CLS.supported_datasets())


def DATASET_TYPE(dataset, *, default: str = 'MCQ') -> str:
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'TYPE'):
                return cls.TYPE
    # Have to add specific routine to handle ConcatDataset
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        TYPES = [DATASET_TYPE(dname) for dname in dataset_list]
        if not np.all([x == TYPES[0] for x in TYPES]):
            warnings.warn(f'Error datasets type, data list: {dataset_list}, typs: {TYPES}')
            return 

        return TYPES[0]

    if 'openended' in dataset.lower():
        return 'VQA'
    warnings.warn(f'Dataset {dataset} is a custom one and not annotated as `openended`, will treat as {default}. ')
    return default


def DATASET_MODALITY(dataset, *, default: str = 'IMAGE') -> str:
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'MODALITY'):
                return cls.MODALITY
    # Have to add specific routine to handle ConcatDataset
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        MODALITIES = [DATASET_MODALITY(dname) for dname in dataset_list]
        if not np.all([x == MODALITIES[0] for x in MODALITIES]):
            warnings.warn(f'Error datasets MODALITIES, data list: {dataset_list}, MODALITIES: {MODALITIES}')
            return 
        return MODALITIES[0]

    if 'VIDEO' in dataset.lower():
        return 'VIDEO'
    elif 'IMAGE' in dataset.lower():
        return 'IMAGE'
    warnings.warn(f'Dataset {dataset} is a custom one, will treat modality as {default}. ')
    return default


def build_dataset(dataset_name,dataset_path):
    try:
        for cls in DATASET_CLASSES:
            if dataset_name in cls.supported_datasets():
                dataset = cls(dataset=dataset_name,dataset_path=dataset_path)
                return dataset
    except:
        return None


__all__ = [
    'build_dataset', 'img_root_map', 'build_judge', 'extract_answer_from_item', 'prefetch_answer', 'DEBUG_MESSAGE'
] + [cls.__name__ for cls in DATASET_CLASSES]
