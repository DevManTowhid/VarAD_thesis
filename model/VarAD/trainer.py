import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tool import calculate_metric, calculate_average_metric, plot_sample_cv2
from .anomaly_mamba import AnomalyMamba


class AnomalyMambaTrainer(nn.Module):
    def __init__(
            self,

            # tokenizer ----------------
            backbone='wide_resnet50_2',
            norm=False,
            hierarchies=[1,2,3],
            # VSSM --------------------
            depths=[2, 2],
            valid_path=0,
            bos_length=4,
            pred_next_n=8,
            lambda_spatial=1.,
            learnable_bos=False,
            use_pe=False,
            is_proj=True,
            use_path_weights=False,
            # training-related  -------
            image_size=256,
            device='cuda:0',
            learning_rate=1e-3,
            pretrained=None,
            adapter=['linear'],

    ):

        super(AnomalyMambaTrainer, self).__init__()

        self.device = device
        cache_data = torch.zeros((1,3,image_size,image_size))

        self.model = AnomalyMamba(
            pred_next_n=pred_next_n,
            backbone=backbone,
            hierarchies=hierarchies,
            norm=norm,
            depths=depths,
            bos_length=bos_length,
            valid_path=valid_path,
            lambda_spatial=lambda_spatial,
            image_size=image_size,
            learnable_bos=learnable_bos,
            use_pe=use_pe,
            is_proj=is_proj,
            cache_data=cache_data,
            use_path_weights=use_path_weights,
            adapter=adapter
        )
        self.model.to(device)

        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        self.image_size = image_size
        self.data_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])

        self.target_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()])



        # build the optimizer
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, betas=(0.5, 0.999))

        if pretrained:
            self.load(pretrained)

    @staticmethod
    def setup_paths(args):
        save_root = os.path.join(args.save_path, 'VarAD')
        model_root = os.path.join(save_root, 'models')
        log_root = os.path.join(save_root, 'logs')
        csv_root = os.path.join(save_root, 'csvs')
        image_root = os.path.join(save_root, 'images')
        tensorboard_root = os.path.join(save_root, 'tensorboard')

        os.makedirs(model_root, exist_ok=True)
        os.makedirs(log_root, exist_ok=True)
        os.makedirs(csv_root, exist_ok=True)
        os.makedirs(image_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)

        # prepare model name
        model_name = '{:}-{:}-{:}-{:}-{:}-{:}-{:}'.format(args.dataset, args.model,
                                                       args.image_size, args.pred_next_n,
                                                           args.adapter, args.hierarchies, args.valid_path)

        # prepare model path
        ckp_path = os.path.join(model_root, f'{model_name}-{args.category}')

        # prepare tensorboard dir
        tensorboard_dir = os.path.join(tensorboard_root, f'{model_name}')
        if os.path.exists(tensorboard_dir):
            import shutil
            shutil.rmtree(tensorboard_dir)
        tensorboard_logger = SummaryWriter(log_dir=tensorboard_dir)

        # prepare csv path
        csv_path = os.path.join(csv_root, f'{model_name}.csv')

        # prepare image path
        image_dir = os.path.join(image_root, f'{model_name}')
        os.makedirs(image_dir, exist_ok=True)

        # prepare log path
        log_path = os.path.join(log_root, f'{model_name}.txt')

        return model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger

    # @staticmethod
    # def setup_paths(args):
    #     save_root = os.path.join(args.save_path, 'VarAD')
    #     model_root = os.path.join(save_root, 'models')
    #     log_root = os.path.join(save_root, 'logs')
    #     csv_root = os.path.join(save_root, 'csvs')
    #     image_root = os.path.join(save_root, 'images')
    #     tensorboard_root = os.path.join(save_root, 'tensorboard')
    #
    #     os.makedirs(model_root, exist_ok=True)
    #     os.makedirs(log_root, exist_ok=True)
    #     os.makedirs(csv_root, exist_ok=True)
    #     os.makedirs(image_root, exist_ok=True)
    #     os.makedirs(tensorboard_root, exist_ok=True)
    #
    #     # prepare model name
    #     model_name = '{:}-{:}-{:}-{:}-PROJ-{:}-{:}'.format(args.dataset, args.model,
    #                                                    args.image_size, args.pred_next_n,
    #                                                        args.is_proj, args.hierarchies)
    #
    #     # prepare model path
    #     ckp_path = os.path.join(model_root, f'{model_name}-{args.category}')
    #
    #     # prepare tensorboard dir
    #     tensorboard_dir = os.path.join(tensorboard_root, f'{model_name}')
    #     if os.path.exists(tensorboard_dir):
    #         import shutil
    #         shutil.rmtree(tensorboard_dir)
    #     tensorboard_logger = SummaryWriter(log_dir=tensorboard_dir)
    #
    #     # prepare csv path
    #     csv_path = os.path.join(csv_root, f'{model_name}.csv')
    #
    #     # prepare image path
    #     image_dir = os.path.join(image_root, f'{model_name}')
    #     os.makedirs(image_dir, exist_ok=True)
    #
    #     # prepare log path
    #     log_path = os.path.join(log_root, f'{model_name}.txt')
    #
    #     return model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger

    def save(self, path):
        self.save_dict = {}
        for param, value in self.state_dict().items():
            # if 'frozen_tokenizer' in param:
            #     continue
            self.save_dict[param] = value

        torch.save(self.save_dict, path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def train_epoch(self, loader):
        self.model.train()
        loss_list = []
        for idx, items in enumerate(loader):
            data = items['img']
            data = data.to(self.device)

            patch_embed, predict_embed = self.model(data)
            loss = self.model.cal_loss(patch_embed, predict_embed)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())

        return np.mean(loss_list)

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def compute_fps(self, dataloader):


        # warm up
        global_image = None
        for indx, items in enumerate(dataloader):
            image = items['img'].to(self.device)
            # pixel level
            patch_embed, predict_embed = self.model(image)
            anomaly_map = self.model.cal_am(patch_embed, predict_embed)
            global_image = image

        global_image = global_image[0:1, :, :, :]
        time_begin = time.time()
        image_number = 0
        for _ in range(100):
            # pixel level
            patch_embed, predict_embed = self.model(global_image)
            anomaly_map = self.model.cal_am(patch_embed, predict_embed)
            image_number += global_image.shape[0]

        time_end = time.time()
        from thop import profile
        flops, params = profile(self.model, (global_image, ))
        print('flops: %.2f M, oarams: %.2f M'%(flops/1000000.0, params/1000000.0))
        fps = image_number / (time_end - time_begin)
        return fps

    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None, cal_pro=False, by_specie=False):
        self.model.eval()
        results = {}
        results['cls_names'] = []
        results['imgs_gts'] = []
        results['anomaly_scores'] = []
        results['imgs_masks'] = []
        results['anomaly_maps'] = []
        results['imgs'] = []
        results['names'] = []
        results['specie_name'] = []

        image_indx = 0
        for indx, items in enumerate(dataloader):
            # extract meta data
            path = items['img_path']
            specie_name = items['specie_name']
            cls_name = items['cls_name']
            for _cls_name, _specie_name in zip(cls_name, specie_name):
                image_indx += 1
                results['names'].append('{:}-{:}-{:03d}'.format(_cls_name, _specie_name, image_indx))
                results['specie_name'].append(_specie_name)

            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

            for _gt_mask in gt_mask:
                results['imgs_masks'].append(_gt_mask.squeeze(0).numpy())
            is_anomaly = np.array(items['anomaly'])
            for _is_anomaly in is_anomaly:
                results['imgs_gts'].append(_is_anomaly)

            if save_fig:
                for _path in path:
                    vis_image = cv2.resize(cv2.imread(_path), (self.image_size, self.image_size))
                    results['imgs'].append(vis_image)

            image = items['img'].to(self.device)
            cls_name = items['cls_name']
            results['cls_names'].extend(cls_name)

            # pixel level
            patch_embed, predict_embed = self.model(image)
            anomaly_map = self.model.cal_am(patch_embed, predict_embed)
            anomaly_score = [np.max(s) for s in anomaly_map]

            for _anomaly_map, _anomaly_score in zip(anomaly_map, anomaly_score):
                results['anomaly_maps'].append(_anomaly_map)
                results['anomaly_scores'].append(_anomaly_score)

        # visualization
        if save_fig:
            print('saving fig.....')
            plot_sample_cv2(
                results['names'],
                results['imgs'],
                {'mamba': results['anomaly_maps']},
                results['imgs_masks'],
                save_fig_dir,norm_by_sample=True
            )

        # metrics
        if by_specie:
            specie_set = sorted(list(set(results['specie_name'])))

        metric_dict = dict()
        for obj in obj_list:
            if by_specie:
                for specie in specie_set:
                    if specie == 'good':
                        continue
                    else:
                        metric_dict[f'{obj}-{specie}'] = dict()
            else:
                metric_dict[obj] = dict()

        for obj in obj_list:
            if by_specie:
                for specie in specie_set:
                    if specie == 'good':
                        continue
                    else:
                        metric = calculate_metric(results,obj, cal_pro, by_specie, specie)
                        obj_full_name = f'{obj}-{specie}'
                        metric_dict[obj_full_name] = metric
            else:
                metric = calculate_metric(results, obj, cal_pro)
                # metric = calculate_metric(results, obj, True)
                obj_full_name = f'{obj}'
                metric_dict[obj_full_name] = metric

        metric_dict['Average'] = calculate_average_metric(metric_dict)

        return metric_dict
