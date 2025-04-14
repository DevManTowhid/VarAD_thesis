import pandas as pd
import os
import logging

############ for csv files
def write_results(results: dict, current_class, total_classes, csv_path):
    keys = list(results.keys())

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            row_data = {k: 0.00 for k in keys}
            df_temp = pd.DataFrame(row_data, index=[class_name])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys:
        df.loc[current_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')


def save_metric(metrics, total_classes, class_name, csv_path):
    total_classes_processed = []
    for indx, k in enumerate(total_classes):
        if k.isdigit():
            total_classes_processed.append(f'b{total_classes[indx]}')
        else:
            total_classes_processed.append(f'{total_classes[indx]}')

    if class_name.isdigit():
        class_name_processed = f'b{class_name}'
    else:
        class_name_processed = f'{class_name}'

    write_results(metrics, class_name_processed, total_classes_processed, csv_path)



class Logger(object):
    def __init__(self, txt_path):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.WARNING)
        self.txt_path = txt_path
        self.logger = logging.getLogger('train')
        self.formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        self.logger.setLevel(logging.INFO)

    def __console(self, level, message):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        file_handler = logging.FileHandler(self.txt_path, mode='a')
        console_handler = logging.StreamHandler()

        file_handler.setFormatter(self.formatter)
        console_handler.setFormatter(self.formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)

        self.logger.removeHandler(file_handler)
        self.logger.removeHandler(console_handler)

        file_handler.close()

    def debug(self, message):
        self.__console('debug', message)

    def info(self, message):
        self.__console('info', message)

    def warning(self, message):
        self.__console('warning', message)

    def error(self, message):
        self.__console('error', message)


def log_metrics(metrics, logger, tensorboard_logger, epoch):
    def log_single_class(data, tag):
        logger.info(
            '{:>15} \t\tI-Auroc:{:.2f} \tI-F1:{:.2f} \tI-AP:{:.2f} \tP-Auroc:{:.2f} \tP-F1:{:.2f} \tP-AP:{:.2f} \tPRO:{:.2f} \tmax-IoU:{:.2f}'.
            format(tag,
                   data['auroc_im'],
                   data['f1_im'],
                   data['ap_im'],
                   data['auroc_px'],
                   data['f1_px'],
                   data['ap_px'],
                   data['aupro'],
                   data['max_iou'],
                   )
        )
        # Adding scalar metrics to TensorBoard
        for metric_name in ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px', 'aupro']:
            tensorboard_logger.add_scalar(f'{tag}-{metric_name}', data[metric_name], epoch)

    for tag, data in metrics.items():
        log_single_class(data, tag)