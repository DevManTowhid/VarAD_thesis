# function1: extract desired metrics for specific names regarding a dataset list
import os
import pandas as pd

def read_csv(root_dir, sub_dir, dataset_list, name_formatting:str, required_metrics, return_average:bool):
    def _compute_average(df: pd.DataFrame):
        filtered_df = df[(df.index != 'Average') & (df.sum(axis=1) != 0)]
        avg_values = filtered_df.mean(axis=0)
        return avg_values

    path_list = []

    ######## dataset, category name
    for dataset_name in dataset_list:
        full_path = os.path.join(root_dir, sub_dir, name_formatting.format(dataset_name))
        path_list.append(full_path)

    if return_average:
        df_list = []
        for path, dataset_name in zip(path_list, dataset_list):
            try:
                df = pd.read_csv(path, index_col=0)

                avg_values = _compute_average(df)
                # avg_df = {dataset_name: avg_values}
                df_list.append(avg_values)

            except Exception as e:
                print(f'Error in reading {path} for {dataset_name}: {str(e)}')
        result_df = pd.DataFrame(df_list, index=dataset_list)
    else:
        df_list = []
        for path, dataset_name in zip(path_list, dataset_list):
            try:
                df = pd.read_csv(path, index_col=0)
                df_list.append(df)
            except Exception as e:
                print(f'Error in reading {path} for {dataset_name}: {str(e)}')

        result_df = pd.concat(df_list, axis=0)
    result_df = result_df.loc[:, required_metrics]

    return result_df


if __name__ == '__main__':

    ## a short example
    # csv_path = ['../test/mamba.csv', '../test/cdo.csv']
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = ['VarAD/csvs', 'CDO/csvs']
    # dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    # name_formatting = ['{:}-DINO-252-64.csv', '{:}-hrnet32-512.csv']
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px', 'auroc_im', 'f1_im', 'ap_im']
    # return_average = False
    # for _csv_path, _sub_dir, _name_formatting in zip(csv_path, sub_dir, name_formatting):
    #     df = read_csv(root_dir, _sub_dir, dataset_list, _name_formatting, required_metrics, return_average)
    #     df.to_csv(_csv_path, header=True, float_format='%.2f')

    # ## for feature adapters
    # csv_path = '../test/mamba_{:}.csv'
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = 'VarAD/csvs'
    # dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    # prefix_list = ['True', 'False']
    # name_formatting = 'DINO-1022-256-PROJ-{:}-[1, 2, 3].csv'
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px']
    # return_average = True
    # for prefix in prefix_list:
    #     name = '{:}-' + name_formatting.format(prefix)
    #     df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
    #     df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    # ## for backbones
    # csv_path = '../test/mamba_{:}.csv'
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = 'VarAD/csvs'
    # dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    # prefix_list = ['dinov2_vitb14', 'dinov2_vitl14']
    # name_formatting = '{:}-1022-256-PROJ-True-[1, 2, 3].csv'
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px']
    # return_average = True
    # for prefix in prefix_list:
    #     name = '{:}-' + name_formatting.format(prefix)
    #     df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
    #     df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    ## for paths
    # csv_path = '../test/mamba_path_{:}.csv'
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = 'VarAD/csvs'
    # dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    # prefix_list = [0, 1, 2, 3]
    # name_formatting = 'DINO-1022-256-PROJ-True-[2, 3]-{:}.csv'
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px']
    # return_average = True
    # for prefix in prefix_list:
    #     name = '{:}-' + name_formatting.format(prefix)
    #     df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
    #     df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    # backbone
    # csv_path = '../test/mamba_backbone_{:}.csv'
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = 'VarAD/csvs'
    # dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    # prefix_list = ['CLIP', 'MAE', 'VIT']
    #
    # name_formatting = "{:}-1024-256-['linear']-[1, 2, 3]--1.csv"
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px']
    # return_average = True
    # for prefix in prefix_list:
    #     name = '{:}-' + name_formatting.format(prefix)
    #     df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
    #     df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    ###### adapter
    # csv_path = '../test/mamba_adapter_{:}.csv'
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = 'VarAD/csvs'
    # dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    # prefix_list = [['resblock'], ['linear', 'lora'], ['resblock', 'lora'], ['lora']]
    # # prefix_list = [['resblock'], ['linear', 'lora'], ['resblock', 'lora']]
    #
    # name_formatting = "DINO-1022-256-{:}-[1, 2, 3]--1.csv"
    # # btad-DINO-1022-256-{:}-[1, 2, 3]--1.csv
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px']
    # return_average = True
    # for prefix in prefix_list:
    #     name = '{:}-' + name_formatting.format(prefix)
    #     df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
    #     df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    ## for different methods
    csv_path = '../test/mamba_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'VarAD/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['252-64', '504-128', '756-192', '1022-256']
    name_formatting = 'DINO-{:}-PROJ-True-[1, 2, 3].csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    csv_path = '../test/cdo_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'CDO/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['256', '512', '768', '1024']
    name_formatting = 'hrnet32-{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')


    csv_path = '../test/PyramidFlow_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'PyramidFlow/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['256', '512', '768', '1024']
    name_formatting = '{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    csv_path = '../test/PFM_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'PFM/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['256', '512', '768', '1024']
    name_formatting = '{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    csv_path = '../test/AMI_Net_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'AMI_Net/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['256', '512', '768', '1024']
    name_formatting = '{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    csv_path = '../test/PNPT_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'PNPT/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['256', '512', '768', '1024']
    name_formatting = '{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    csv_path = '../test/MSFlow_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'MSFlow/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['256', '512', '768', '1024']
    name_formatting = '{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    csv_path = '../test/MSFlow_button_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'MSFlow/csvs'
    dataset_list = ['button']
    prefix_list = ['1024']
    name_formatting = '{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = False
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')


    csv_path = '../test/PatchCore_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'PatchCore/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['256', '512', '768', '1024']
    name_formatting = 'wide_resnet50_2-{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')
    #
    #
    csv_path = '../test/RD4AD_{:}.csv'
    root_dir = '/home/anyad/AnomalyDetection/workspace'
    sub_dir = 'RD4AD/csvs'
    dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    prefix_list = ['256', '512', '768', '1024']
    name_formatting = '{:}.csv'
    required_metrics = ['auroc_im', 'f1_im', 'ap_im', 'auroc_px', 'f1_px', 'ap_px']
    return_average = True
    for prefix in prefix_list:
        name = '{:}-' + name_formatting.format(prefix)
        df = read_csv(root_dir, sub_dir, dataset_list, name, required_metrics, return_average)
        df.to_csv(csv_path.format(prefix), header=True, float_format='%.2f')

    # ### for different steps
    # csv_path = '../test/mamba_step_{step}.csv'
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = 'VarAD/csvs'
    # dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    # # name_formatting = '{:}-DINO-1022-{step}.csv'
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px']
    # return_average = True
    #
    # step_list = [1024, 512, 256, 64]
    #
    # for step in step_list:
    #     'mvtec-DINO-1022-512-PROJ-True-[1, 2, 3].csv'
    #     name_formatting = f'DINO-1022-{step}-PROJ-True-[1, 2, 3].csv'
    #     name_formatting = '{:}-' + name_formatting
    #     df = read_csv(root_dir, sub_dir, dataset_list, name_formatting, required_metrics, return_average)
    #     df.to_csv(csv_path.format(step=step), header=True, float_format='%.2f')
    #
    # ### for different steps
    # csv_path = '../test/mamba_hierarchy_{:}.csv'
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = 'VarAD/csvs'
    # dataset_list = ['mvtec', 'visa', 'btad', 'dtd']
    # # name_formatting = '{:}-DINO-1022-{step}.csv'
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px']
    # return_average = True
    #
    # hierarchy_list = [[1], [2], [3], [1,2],[2,3,], [1,2,3]]
    #
    # for hierarchy in hierarchy_list:
    #     'mvtec-DINO-1022-512-PROJ-True-[1, 2, 3].csv'
    #     name_formatting = f'DINO-1022-256-PROJ-True-{hierarchy}.csv'
    #     name_formatting = '{:}-' + name_formatting
    #     df = read_csv(root_dir, sub_dir, dataset_list, name_formatting, required_metrics, return_average)
    #     df.to_csv(csv_path.format(hierarchy), header=True, float_format='%.2f')


    ### for button
    # csv_path = ['../test/mamba_button.csv',
    #             '../test/cdo_button.csv',
    #             '../test/pfm_button.csv',
    #             '../test/pyramidflow_button.csv',
    #             '../test/patchcore_button.csv',
    #             '../test/rd4ad_button.csv',
    #             ]
    #
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    # sub_dir = ['VarAD/csvs',
    #            'CDO/csvs',
    #            'PFM/csvs',
    #            'PyramidFlow/csvs',
    #            'PatchCore/csvs',
    #            'RD4AD/csvs'
    #            ]
    #
    # dataset_list = ['button']
    # name_formatting = ['{:}-dinov2_vits14-1022-256-PROJ-True-[2, 3]--1.csv',
    #                    '{:}-hrnet32-1024.csv',
    #                    '{:}-1024.csv',
    #                    '{:}-1024.csv',
    #                    '{:}-wide_resnet50_2-1024.csv',
    #                    '{:}-1024.csv',
    #                    ]
    # required_metrics = ['auroc_px', 'f1_px', 'ap_px']
    #
    # return_average = False
    # for _csv_path, _sub_dir, _name_formatting in zip(csv_path, sub_dir, name_formatting):
    #     df = read_csv(root_dir, _sub_dir, dataset_list, _name_formatting, required_metrics, return_average)
    #     df.to_csv(_csv_path, header=True, float_format='%.2f')