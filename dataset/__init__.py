from .mvtec import MVTEC_CLS_NAMES, MVTecDataset, MVTEC_ROOT
from .visa import VISA_CLS_NAMES, VisaDataset, VISA_ROOT
from .btad import BTAD_CLS_NAMES, BTADDataset, BTAD_ROOT
from .dtd import DTD_CLS_NAMES,DTDDataset,DTD_ROOT

from .base import BaseDataset
from torch.utils.data import ConcatDataset

dataset_dict = {
    'btad': (BTAD_CLS_NAMES, BTADDataset, BTAD_ROOT),
    'dtd': (DTD_CLS_NAMES, DTDDataset, DTD_ROOT),
    'mvtec': (MVTEC_CLS_NAMES, MVTecDataset, MVTEC_ROOT),
    'visa': (VISA_CLS_NAMES, VisaDataset, VISA_ROOT),
}

def get_dataset(dataset_types, transform, target_transform, training, mode, k_shot,
                synthesis_anomalies=False, white_noise=False, class_name_list=None):
    """
    Load datasets and prepare for training or testing.

    :param dataset_types: List of dataset types to be loaded, e.g., ['mvtec', 'visa']
    :param transform: Function for transforming images
    :param target_transform: Function for transforming ground truths
    :param training: Boolean, True for training, False for testing
    :param mode: String, 'ZS' for zero-shot learning, 'FS' for few-shot learning
    :param k_shot: Integer, number of normal samples (works only for few-shot)
    :param synthesis_anomalies: Boolean, whether to synthesize anomalies in training
    :param class_name_list: List of class names to load specific categories from datasets
    :return: Tuple containing class names, dataset instances, and dataset roots
    """
    if not isinstance(dataset_types, list):
        dataset_types = [dataset_types]

    dataset_class_names_list = []
    dataset_instances_list = []
    dataset_roots_list = []

    for indx, dataset_type in enumerate(dataset_types):
        if dataset_dict.get(dataset_type, ''):
            dataset_class_names, dataset_instance, dataset_root = dataset_dict[dataset_type]
            if class_name_list is not None:
                dataset_class_names = class_name_list[indx]

            print(dataset_class_names)

            dataset_instance = dataset_instance(
                class_names=dataset_class_names,
                transform=transform,
                target_transform=target_transform,
                training=training,
                mode=mode,
                k_shot=k_shot,
                synthesis_anomalies=synthesis_anomalies,
                white_noise=white_noise
            )

            dataset_class_names_list.append(dataset_class_names)
            dataset_instances_list.append(dataset_instance)
            dataset_roots_list.append(dataset_root)

        else:
            print(f'Only support {dataset_dict.keys()}, but entered {dataset_type}...')
            raise NotImplementedError

    if len(dataset_types) > 1:
        dataset_instance = ConcatDataset(dataset_instances_list)
        dataset_class_names = dataset_class_names_list
        dataset_root = dataset_roots_list
    else:
        dataset_instance = dataset_instances_list[0]
        dataset_class_names = dataset_class_names_list[0]
        dataset_root = dataset_roots_list[0]

    return dataset_class_names, dataset_instance, dataset_root
