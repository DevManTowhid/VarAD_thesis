from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
# Importing from local modules
from dataset import get_dataset
from model.VarAD.trainer import AnomalyMambaTrainer
from tool import save_metric, setup_seed, Logger, log_metrics


setup_seed(111)


def train(args):
    # Configurations
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up paths
    model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger = AnomalyMambaTrainer.setup_paths(args)

    # Extract configurations
    save_fig = args.save_fig
    do_train = args.do_train
    cal_pro = args.cal_pro

    # Logger
    logger = Logger(log_path)

    # Print basic information
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    logger.info('Model name: {:}'.format(model_name))

    model = AnomalyMambaTrainer(
        backbone=args.model,
        norm=True,
        depths=[1],
        hierarchies=args.hierarchies,
        # bos_length=64,
        pred_next_n=args.pred_next_n,
        valid_path=args.valid_path,
        image_size=args.image_size,
        learning_rate=learning_rate,
        device=device,
        pretrained=args.ckt_path,
        is_proj=args.is_proj,
        adapter=args.adapter
    )

    train_data_cls_names, train_data, train_data_root = get_dataset(
        dataset_types=args.dataset,
        transform=model.data_transform,
        target_transform=model.target_transform,
        training=True, mode='FS', k_shot=1e6, synthesis_anomalies=False, class_name_list=[[args.category]])

    test_data_cls_names, test_data, test_data_root = get_dataset(
        dataset_types=args.dataset,
        transform=model.data_transform,
        target_transform=model.target_transform,
        training=False, mode='FS', k_shot=0, synthesis_anomalies=False, class_name_list=[[args.category]])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    if args.compute_fps:
        torch.backends.cudnn.benchmark = True
        fps_test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=4)
        fps = model.compute_fps(fps_test_dataloader)
        print(f'fps: {fps:.2f}')

        return

    best_f1 = 0
    by_specie = (args.dataset == 'mvtec_loco')
    # by_specie = False

    if do_train:
        for epoch in tqdm(range(epochs)):
            loss = model.train_epoch(train_dataloader)

            # Logs
            if (epoch + 1) % args.print_freq == 0:
                logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss))
                tensorboard_logger.add_scalar('loss', loss, epoch)

            # Validation
            if (epoch + 1) % args.valid_freq == 0 or (epoch == epochs - 1):
                if epoch == epochs - 1:
                    save_fig_flag = save_fig
                else:
                    save_fig_flag = False

                # save_fig_flag = True
                logger.info('=============================Testing ====================================')
                metric_dict_novel = model.evaluation(
                    test_dataloader,
                    test_data_cls_names,
                    save_fig_flag,
                    image_dir,
                    cal_pro,
                    by_specie
                )


                log_metrics(
                    metric_dict_novel,
                    logger,
                    tensorboard_logger,
                    epoch
                )

                f1_px = metric_dict_novel['Average']['auroc_px']

                # Save best
                if f1_px > best_f1:
                    for k in metric_dict_novel.keys():
                        if k == 'Average':
                            continue
                        save_metric(metric_dict_novel[k], test_data_cls_names, k, csv_path)

                    ckp_path_best = ckp_path + '_best.pth'
                    model.save(ckp_path_best)
                    best_f1 = f1_px

    else:
        if args.ckt_path:
            ckt_path = args.ckt_path
            logger.info(f'use manually specified ckt path: {ckt_path}')
        else:
            ckt_path = ckp_path + '_best.pth'
            logger.info(f'automatically generate ckt path: {ckt_path}')

        logger.info(f'trying to load {ckt_path}')
        assert os.path.isfile(ckt_path), 'Please train first'
        model.load(ckt_path)

        save_fig_flag = save_fig
        # save_fig_flag = True
        logger.info('=============================Testing ====================================')

        metric_dict = model.evaluation(
            test_dataloader,
            test_data_cls_names,
            save_fig_flag,
            image_dir,
            cal_pro,
            by_specie
        )

        log_metrics(
            metric_dict,
            logger,
            tensorboard_logger,
            epochs
        )

        for k in metric_dict.keys():
            save_metric(metric_dict[k], test_data_cls_names, k, csv_path)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyMamba", add_help=True)
    # Paths and configurations
    parser.add_argument("--dataset", type=str, default="mvtec",   help="Dataset for training")
    parser.add_argument("--category", type=str, default="bottle", help="Category for training")

    parser.add_argument("--save_path", type=str, default='./workspace', help='Path to save results')


    parser.add_argument("--save_fig", type=str2bool, default=False)
    parser.add_argument("--compute_fps", type=str2bool, default=False)
    parser.add_argument("--do_train", type=str2bool, default=True)
    parser.add_argument("--cal_pro", type=str2bool, default=False)
    parser.add_argument("--ckt_path", type=str, default='')

    parser.add_argument("--pred_next_n", type=int, default=256, help="Image size")
    parser.add_argument("--is_proj", type=str2bool, default=True)
    parser.add_argument("--valid_path", type=int, default=-1)

    parser.add_argument("--hierarchies", type=int, nargs='+', default=[1, 2, 3], help="Epochs")
    parser.add_argument("--adapter", type=str, nargs='+', default=['linear'], help="Epochs") # linear resblock lora

    parser.add_argument("--model", type=str, default="dinov2_vits14")

    # Hyper-parameters
    parser.add_argument("--exp_indx", type=int, default=0, help="Epochs")
    parser.add_argument("--epoch", type=int, default=10, help="Epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--print_freq", type=int, default=1, help="Print frequency")
    parser.add_argument("--valid_freq", type=int, default=5, help="Validation frequency")

    args = parser.parse_args()

    if args.model.count('DINO') > 0 or args.model.count('dinov2') > 0:
        args.image_size = 14 * (args.image_size  // 14)


    train(args)
