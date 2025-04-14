import json
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
import glob
import imgaug.augmenters as iaa
from config import DATA_ROOT
import torch
from PIL import Image
from .synthesis_utils import rand_perlin_2d_np

class BaseSolver(object):

    def __init__(self, root, clsnames):
        self.root = root
        self.CLSNAMES = clsnames
        self.path = f'{root}/meta.json'

    def run(self):
        with open(self.path, 'r') as f:
            info = json.load(f)

        info_required = dict(train={}, test={})
        for cls in self.CLSNAMES:
            for k in info.keys():
                info_required[k][cls] = info[k][cls]

        return info_required


class BaseDataset(data.Dataset):
    def __init__(self, class_names, transform, target_transform,
                 root, training, mode, k_shot, synthesis_anomalies=False,
                 white_noise=False, anomaly_source_path=os.path.join(DATA_ROOT, os.path.join('dtd', 'images'))):
        assert mode in ['FS', 'ZS']

        self.mode = mode
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.k_shot = k_shot
        self.data_all = []

        self.solver = BaseSolver(root, class_names)
        meta_info = self.solver.run()
        self.training = training

        self.white_noise = white_noise # for cdo
        if self.mode == 'ZS':
            meta_info = meta_info['test']
        else:
            meta_info = self._select_normal_from_train(meta_info['train'], k_shot) if training \
                else meta_info['test']

            if meta_info == 0:
                print('There are no normal samples in the training dataset. please check.')
                return 0

        self.meta_info = meta_info
        self.class_names = class_names

        for cls_name in self.class_names:
            self.data_all.extend(meta_info[cls_name])

        self.count_number(self.data_all)
        
        self.length = len(self.data_all)
        self.anomaly_source_paths = sorted(glob.glob(os.path.join(anomaly_source_path, "*/*.jpg")))
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]

        self.synthesis_anomalies = synthesis_anomalies

    def _select_normal_from_train(self, meta_info: dict, k_shot):
        normal_info = {}
        for cls in meta_info.keys():
            normal_info[cls] = []
            for info in meta_info[cls]:
                anomaly = info['anomaly']
                if anomaly == 0:
                    normal_info[cls].append(info)

            k = min(len(normal_info[cls]), k_shot)

            if k == 0:
                return 0

            normal_info[cls] = random.sample(normal_info[cls], k)
        return normal_info

    def count_number(self, data):
        normal_number = 0
        abnormal_number = 0
        for _data in data:
            img_path, mask_path, cls_name, specie_name, anomaly = _data['img_path'], _data['mask_path'], _data['cls_name'], \
                                                                  _data['specie_name'], _data['anomaly']

            if anomaly == 0:
                normal_number += 1
            else:
                abnormal_number += 1

        print(f'Training: {self.training}, Normal: {normal_number}, Abnormal: {abnormal_number}')

    def combine_images(self, cls_name):
        img_info = random.sample(self.meta_info[cls_name], 4)

        img_ls = []
        mask_ls = []

        for data in img_info:
            img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                                  data['specie_name'], data['anomaly']
            img_path = os.path.join(self.root, img_path)
            mask_path = os.path.join(self.root, mask_path)

            img = Image.open(img_path)
            img_ls.append(img)

            if not anomaly:
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

            mask_ls.append(img_mask)

        result_image = self._combine_images_helper(img_ls)
        result_mask = self._combine_images_helper(mask_ls)

        return result_image, result_mask

    @staticmethod
    def _combine_images_helper(image_list):
        image_width, image_height = image_list[0].size
        result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
        for i, img in enumerate(image_list):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_image.paste(img, (x, y))
        return result_image

    def augment_image_white_noise(self, image):

        # generate noise image
        noise_image = np.random.randint(0, 255, size=image.shape).astype(np.float32) / 255.0
        patch_mask = np.zeros(image.shape[:2], dtype=np.float32)


        # generate random mask
        patch_number = np.random.randint(0, 5)
        augmented_image = image

        MAX_TRY_NUMBER = 200
        for i in range(patch_number):
            try_count = 0
            coor_min_dim1 = 0
            coor_min_dim2 = 0

            coor_max_dim1 = 0
            coor_max_dim2 = 0
            while try_count < MAX_TRY_NUMBER:
                try_count += 1

                patch_dim1 = np.random.randint(image.shape[0] // 40, image.shape[0] // 10)
                patch_dim2 = np.random.randint(image.shape[1] // 40, image.shape[1] // 10)

                center_dim1 = np.random.randint(patch_dim1, image.shape[0] - patch_dim1)
                center_dim2 = np.random.randint(patch_dim2, image.shape[1] - patch_dim2)

                coor_min_dim1 = np.clip(center_dim1 - patch_dim1, 0, image.shape[0])
                coor_min_dim2 = np.clip(center_dim2 - patch_dim2, 0, image.shape[1])

                coor_max_dim1 = np.clip(center_dim1 + patch_dim1, 0, image.shape[0])
                coor_max_dim2 = np.clip(center_dim2 + patch_dim2, 0, image.shape[1])

                break

            patch_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] = 1.0

        augmented_image[patch_mask > 0] = noise_image[patch_mask > 0]

        patch_mask = patch_mask[:, :, np.newaxis]

        if patch_mask.max() > 0:
            has_anomaly = 1.0
        else:
            has_anomaly = 0.0

        return augmented_image, patch_mask, np.array([has_anomaly], dtype=np.float32)

    def augment_image(self, image, anomaly_source_path):
        aug = self._rand_augmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(image.shape[1], image.shape[0]))
        anomaly_img_augmented = aug(image=anomaly_source_img)

        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def _rand_augmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def transform_image(self, image: Image):
        image = image.resize((1024, 1024))
        anomaly_source_path = random.choice(self.anomaly_source_paths)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = np.array(image).astype(np.float32) / 255.0
        if not self.white_noise:
            augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        else:
            augmented_image, anomaly_mask, has_anomaly = self.augment_image_white_noise(image)
        augmented_image = augmented_image * 255.0
        augmented_image = augmented_image.astype(np.uint8)

        image = (image * 255.0).astype(np.uint8)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        augmented_image = Image.fromarray(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        anomaly_mask = Image.fromarray(anomaly_mask[:, :, 0].astype(np.uint8) * 255, mode='L')

        return image, augmented_image, anomaly_mask, has_anomaly

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        return_dict = {
            'specie_name': specie_name,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': os.path.join(self.root, img_path)
        }

        if self.training and self.synthesis_anomalies:

            image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(img)

            augmented_image = self.transform(augmented_image) if self.transform is not None else augmented_image
            anomaly_mask = self.target_transform(
                anomaly_mask) if self.target_transform is not None and anomaly_mask is not None else anomaly_mask
            anomaly_mask = [] if anomaly_mask is None else anomaly_mask

            return_dict.update({
                'augmented_image': augmented_image,
                'anomaly_mask': anomaly_mask,
                'has_anomaly': has_anomaly
            })

        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        return_dict.update({
            'img': img,
            'img_mask': img_mask
        })

        return return_dict

#
# class BaseDataset(data.Dataset):
#     def __init__(self, clsnames, transform, target_transform, aug_rate,
#                  root, training, mode, k_shot, synthesis_anomalies=False,
#                  anomaly_source_path=os.path.join(DATA_ROOT, os.path.join('dtd', 'images'))):
#         assert mode in ['FS', 'ZS']
#         # in zero-shot, the training set should from the testing set
#         # in few-shot, the training set should from the training set
#         self.mode = mode
#         self.root = root
#         self.transform = transform
#         self.target_transform = target_transform
#         self.aug_rate = aug_rate
#         self.k_shot = k_shot
#
#         self.data_all = []
#
#         self.solver = BaseSolver(root, clsnames)
#         meta_info = self.solver.run()
#         self.training = training
#
#         if self.mode == 'ZS':
#             meta_info = meta_info['test']
#         else:
#             if training:
#                 temp = self.select_normal_from_train(meta_info['train'], k_shot)
#                 if temp == 0:
#                     temp = self.select_normal_from_train(meta_info['test'], k_shot)
#                 meta_info = temp
#             else:
#                 meta_info = meta_info['test']
#
#         self.meta_info = meta_info
#         self.cls_names = clsnames
#
#         for cls_name in self.cls_names:
#             self.data_all.extend(meta_info[cls_name])
#
#         self.length = len(self.data_all)
#
#         self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))
#
#         self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
#                       iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
#                       iaa.pillike.EnhanceSharpness(),
#                       iaa.AddToHueAndSaturation((-50,50),per_channel=True),
#                       iaa.Solarize(0.5, threshold=(32,128)),
#                       iaa.Posterize(),
#                       iaa.Invert(),
#                       iaa.pillike.Autocontrast(),
#                       iaa.pillike.Equalize(),
#                       iaa.Affine(rotate=(-45, 45))
#                       ]
#
#         self.synthesis_anomalies = synthesis_anomalies
#
#     def select_normal_from_train(self, meta_info: dict, k_shot):
#         normal_info = {}
#         for cls in meta_info.keys():
#             normal_info[cls] = []
#             for info in (meta_info[cls]):
#                 anomaly = info['anomaly']
#                 if anomaly == 0:
#                     normal_info[cls].append(info)
#
#             k = min(len(normal_info[cls]), k_shot)
#
#             if k == 0:
#                 return 0
#
#             normal_info[cls] = random.sample(normal_info[cls], k)
#         return normal_info
#
#     def combine_img(self, cls_name):
#         img_info = random.sample(self.meta_info[cls_name], 4)
#
#         img_ls = []
#         mask_ls = []
#
#         for data in img_info:
#             img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
#                                                                   data['specie_name'], data['anomaly']
#             img_path = os.path.join(self.root, img_path)
#             mask_path = os.path.join(self.root, mask_path)
#
#             img = Image.open(img_path)
#             img_ls.append(img)
#             if not anomaly:
#                 img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
#             else:
#                 img_mask = np.array(Image.open(mask_path).convert('L')) > 0
#                 img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
#
#             mask_ls.append(img_mask)
#
#         # image
#         image_width, image_height = img_ls[0].size
#         result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
#         for i, img in enumerate(img_ls):
#             row = i // 2
#             col = i % 2
#             x = col * image_width
#             y = row * image_height
#             result_image.paste(img, (x, y))
#
#         # mask
#         result_mask = Image.new("L", (2 * image_width, 2 * image_height))
#         for i, img in enumerate(mask_ls):
#             row = i // 2
#             col = i % 2
#             x = col * image_width
#             y = row * image_height
#             result_mask.paste(img, (x, y))
#
#         return result_image, result_mask
#
#     def augment_image(self, image, anomaly_source_path):
#         aug = self.randAugmenter()
#         perlin_scale = 6
#         min_perlin_scale = 0
#         anomaly_source_img = cv2.imread(anomaly_source_path)
#         anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(image.shape[1], image.shape[0]))
#
#         anomaly_img_augmented = aug(image=anomaly_source_img)
#         perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
#         perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
#
#         perlin_noise = rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))
#         threshold = 0.5
#         perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
#         perlin_thr = np.expand_dims(perlin_thr, axis=2)
#
#         img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
#
#         beta = torch.rand(1).numpy()[0] * 0.8
#
#         augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
#             perlin_thr)
#
#         no_anomaly = torch.rand(1).numpy()[0]
#         if no_anomaly > 0.5:
#             image = image.astype(np.float32)
#             return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
#         else:
#             augmented_image = augmented_image.astype(np.float32)
#             msk = (perlin_thr).astype(np.float32)
#             augmented_image = msk * augmented_image + (1 - msk) * image
#             has_anomaly = 1.0
#             if np.sum(msk) == 0:
#                 has_anomaly = 0.0
#             return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)
#
#     def randAugmenter(self):
#         aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
#         aug = iaa.Sequential([self.augmenters[aug_ind[0]],
#                               self.augmenters[aug_ind[1]],
#                               self.augmenters[aug_ind[2]]]
#                              )
#         return aug
#
#     def transform_image(self, image: Image):
#         image = image.resize((1024, 1024))
#         anomaly_source_path = random.choice(self.anomaly_source_paths)
#         image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#         image = np.array(image).astype(np.float32) / 255.0
#         augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
#         augmented_image = augmented_image * 255.0
#         augmented_image = augmented_image.astype(np.uint8)
#
#         image = (image * 255.0).astype(np.uint8)
#         image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         augmented_image = Image.fromarray(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
#         anomaly_mask = Image.fromarray(anomaly_mask[:, :, 0].astype(np.uint8) * 255, mode='L')
#
#         return image, augmented_image, anomaly_mask, has_anomaly
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         data = self.data_all[index]
#         img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
#                                                               data['specie_name'], data['anomaly']
#         img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
#         if anomaly == 0:
#             img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
#         else:
#             img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
#             img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
#
#         return_dict = {}
#
#         return_dict['specie_name'] = specie_name
#         return_dict['cls_name'] = cls_name
#         return_dict['anomaly'] = anomaly
#         return_dict['img_path'] = os.path.join(self.root, img_path)
#
#         if self.training and self.synthesis_anomalies:
#             image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(img)
#
#             augmented_image = self.transform(augmented_image) if self.transform is not None else augmented_image
#             anomaly_mask = self.target_transform(
#                 anomaly_mask) if self.target_transform is not None and anomaly_mask is not None else anomaly_mask
#             anomaly_mask = [] if anomaly_mask is None else anomaly_mask
#
#             return_dict['augmented_image'] = augmented_image
#             return_dict['anomaly_mask'] = anomaly_mask
#             return_dict['has_anomaly'] = has_anomaly
#
#         img = self.transform(img) if self.transform is not None else img
#         img_mask = self.target_transform(
#             img_mask) if self.target_transform is not None and img_mask is not None else img_mask
#         img_mask = [] if img_mask is None else img_mask
#
#         return_dict['img'] = img
#         return_dict['img_mask'] = img_mask
#
#         return return_dict
#
