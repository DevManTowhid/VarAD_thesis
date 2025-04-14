import os
import numpy as np
import cv2
from loguru import logger

def read_images(root_dir, image_name, suffix, image_size=224, border_ratio=0):
    gt_path = os.path.join(root_dir, image_name+'_gt.jpg')
    ori_path = os.path.join(root_dir, image_name+'_ori.jpg')
    img_path = os.path.join(root_dir, image_name+f'_{suffix}.jpg')

    logger.info(f'gt path:{gt_path}')
    logger.info(f'ori path:{ori_path}')
    logger.info(f'score path:{img_path}')

    try:
        gt_image = cv2.imread(gt_path)
        ori_image = cv2.imread(ori_path)
        img = cv2.imread(img_path)

        gt_image = cv2.resize(gt_image, (image_size, image_size))
        ori_image = cv2.resize(ori_image, (image_size, image_size))
        img = cv2.resize(img, (image_size, image_size))

    except:
        logger.error("The path does not exist..")

    if border_ratio == 0:
        pass
    else:
        border_size = int(image_size * border_ratio)
        gt_image = add_black_border(gt_image, border_size)
        ori_image = add_black_border(ori_image, border_size)
        img = add_black_border(img, border_size)

    return ori_image, gt_image,  img

def add_black_border(image, border_size):
    assert len(image.shape) == 3
    image[:border_size, :, :] = 0
    image[-1-border_size:-1, :, :] = 0
    image[:, :border_size, :] = 0
    image[:, -1-border_size:-1, :] = 0

    return image


def concat_images(image_pairs, nrows, gap_ratio=0.1, save_path=None):
    # image_pairs: (ori_list, gt_list, score_list1.... score_listn)

    ori_image_list = image_pairs[0]

    ncols = ((len(ori_image_list) + nrows - 1) // nrows) * len(image_pairs)

    height = ori_image_list[0].shape[0]
    width = ori_image_list[0].shape[1]
    gap_size = int(height * gap_ratio)

    result_width = width * nrows + gap_size * (nrows - 1)
    result_height = height * ncols + gap_size * (ncols - 1)

    result_image = np.ones((result_height, result_width, 3), dtype=int) * 255

    for i in range(len(image_pairs)):
        for j in range(len(ori_image_list)):

            cur_row = j % nrows
            cur_col = (j // nrows) * len(image_pairs) + i

            if cur_row == 0:
                begin_w = 0
            else:
                begin_w = cur_row * (width + gap_size)

            if cur_col == 0:
                begin_h = 0
            else:
                begin_h = cur_col * (height + gap_size)

            result_image[begin_h:begin_h+height, begin_w:begin_w+width, :] = image_pairs[i][j]

    if save_path is None:
        cv2.imwrite("result_image.png", result_image)
    else:
        cv2.imwrite(save_path, result_image)

def plot_qualitative_results(root_dir, image_names, method_dirs, save_dir, save_name, border_ratio = 0.01, gap_ratio = 0.1, image_size = 224, reverse=False):
    os.makedirs(save_dir, exist_ok=True)
    all_category_img = None
    for image_name in image_names:

        category_imgs = None
        for idx in range(0,len(method_dirs)):
            subset_dir = method_dirs[idx]

            if image_name[1] == 'button' and subset_dir[0].count('VarAD') > 0:
                subset_dir[0] = 'VarAD/images/{:}-dinov2_vits14-1022-256-PROJ-True-[2, 3]--1'

            full_dir = os.path.join(root_dir, subset_dir[0].format(image_name[1]))

            img_list = read_images(full_dir, image_name[0], subset_dir[1], image_size, border_ratio)
            if category_imgs is None:
                category_imgs = [img_list[0], img_list[1]]  # ori + gt

            category_imgs.extend(img_list[2:])  # scores

        if not reverse:
            if all_category_img is None:
                all_category_img = [[] for _ in category_imgs]

            for i in range(len(all_category_img)):
                all_category_img[i].append(category_imgs[i])
        else:
            if all_category_img is None:
                all_category_img = []
            all_category_img.append(category_imgs)

    if not reverse:
        concat_images(all_category_img, len(image_names), gap_ratio,
                      save_path=os.path.join(save_dir, f'{save_name}.png'))
    else:
        concat_images(all_category_img, len(method_dirs)+2, gap_ratio,
                      save_path=os.path.join(save_dir, f'{save_name}.png'))


if __name__ == '__main__':
    # different resolutions
    root_dir = '/home/anyad/AnomalyDetection/workspace'

    # varad
    method_dirs = [
        [
            ['VarAD/images/{:}-DINO-252-64-PROJ-True-[1, 2, 3]', 'mamba'],  # mamba
            ['VarAD/images/{:}-DINO-504-128-PROJ-True-[1, 2, 3]', 'mamba'],  # mamba
            ['VarAD/images/{:}-DINO-756-192-PROJ-True-[1, 2, 3]', 'mamba'],  # mamba
            ['VarAD/images/{:}-DINO-1022-256-PROJ-True-[1, 2, 3]', 'mamba'],  # mamba
        ],
        [
            ['RD4AD/images/{:}-256', 'mamba'],  # mamba
            ['RD4AD/images/{:}-512', 'mamba'],  # mamba
            ['RD4AD/images/{:}-768', 'mamba'],  # mamba
            ['RD4AD/images/{:}-1024', 'mamba'],  # mamba
        ],
        [
            ['PatchCore/images/{:}-wide_resnet50_2-256', 'mamba'],  # mamba
            ['PatchCore/images/{:}-wide_resnet50_2-512', 'mamba'],  # mamba
            ['PatchCore/images/{:}-wide_resnet50_2-768', 'mamba'],  # mamba
            ['PatchCore/images/{:}-wide_resnet50_2-1024', 'mamba'],  # mamba
        ],
        [
            ['PyramidFlow/images/{:}-256', 'mamba'],  # mamba
            ['PyramidFlow/images/{:}-512', 'mamba'],  # mamba
            ['PyramidFlow/images/{:}-768', 'mamba'],  # mamba
            ['PyramidFlow/images/{:}-1024', 'mamba'],  # mamba
        ],
        [
            ['MSFlow/images/{:}-256', 'mamba'],  # mamba
            ['MSFlow/images/{:}-512', 'mamba'],  # mamba
            ['MSFlow/images/{:}-768', 'mamba'],  # mamba
            ['MSFlow/images/{:}-1024', 'mamba'],  # mamba
        ],
        [
            ['PFM/images/{:}-256', 'mamba'],  # mamba
            ['PFM/images/{:}-512', 'mamba'],  # mamba
            ['PFM/images/{:}-768', 'mamba'],  # mamba
            ['PFM/images/{:}-1024', 'mamba'],  # mamba
        ],
        [
            ['AMI_Net/images/{:}-256', 'mamba'],  # mamba
            ['AMI_Net/images/{:}-512', 'mamba'],  # mamba
            ['AMI_Net/images/{:}-768', 'mamba'],  # mamba
            ['AMI_Net/images/{:}-1024', 'mamba'],  # mamba
        ],
        [
            ['PNPT/images/{:}-256', 'mamba'],  # mamba
            ['PNPT/images/{:}-512', 'mamba'],  # mamba
            ['PNPT/images/{:}-768', 'mamba'],  # mamba
            ['PNPT/images/{:}-1024', 'mamba'],  # mamba
        ],
        [
            ['CDO/images/{:}-hrnet32-256', 'mamba'],  # mamba
            ['CDO/images/{:}-hrnet32-512', 'mamba'],  # mamba
            ['CDO/images/{:}-hrnet32-768', 'mamba'],  # mamba
            ['CDO/images/{:}-hrnet32-1024', 'mamba'],  # mamba
        ],

    ]
    #
    image_names = [
        # ['cable-bent_wire-116', 'mvtec'],
        # ['cable-bent_wire-119', 'mvtec'],
    # ['metal_nut-color-105', 'mvtec'],
        # ['metal_nut-scratch-012', 'mvtec'],
        # ['pill-scratch-057', 'mvtec'],
    ['cable-missing_cable-150', 'mvtec'],
        # ['cable-bent_wire-119', 'mvtec'],

    ]

    save_name = ['VarAD',
                 'RD4AD',
                 'PatchCore',
                 'PyramidFlow',
                 'MSFlow',
                 'PFM',
                 'AMI_Net',
                 'PNPT',
                 # 'UniAD',
                 'CDO'
                 ]


    save_dir = '/home/anyad/AnomalyDetection/test/different-resolution'

    for method_dir, name in zip(method_dirs, save_name):

        plot_qualitative_results(root_dir=root_dir,
                                 image_names=image_names,
                                 method_dirs=method_dir,
                                 save_dir=save_dir,
                                 save_name=name,
                                 border_ratio = 0.01, gap_ratio = 0.1, image_size = 224, reverse=False)

    ####### main
    root_dir = '/home/anyad/AnomalyDetection/workspace'

    # varad
    method_dirs = [
        ['PFM/images/{:}-1024', 'mamba'],  # mamba
        ['RD4AD/images/{:}-1024', 'mamba'],  # mamba
        ['PatchCore/images/{:}-wide_resnet50_2-1024', 'mamba'],  # mamba
        ['CDO/images/{:}-hrnet32-1024', 'mamba'],  # mamba
        ['PyramidFlow/images/{:}-1024', 'mamba'],  # mamba
        ['MSFlow/images/{:}-1024', 'mamba'],  # mamba
        ['AMI_Net/images/{:}-1024', 'mamba'],
        ['PNPT/images/{:}-1024', 'mamba'],
        # ['VarAD/images/{:}-dinov2_vits14-1022-256-PROJ-True-[2, 3]--1', 'mamba'],  # mamba
        ['VarAD/images/{:}-DINO-1022-256-PROJ-True-[1, 2, 3]', 'mamba'],  # mamba
    ]

    image_names = [
        ['capsule-crack-011', 'mvtec'],
        ['cable-bent_wire-119', 'mvtec'],
        ['metal_nut-scratch-012', 'mvtec'],
        # ['pill-pill_type-070', 'mvtec'],
        ['screw-scratch_neck-144', 'mvtec'],
        # ['zipper-fabric_interior-124', 'mvtec'],

        ['cashew--083', 'visa'],
        ['pcb1--146', 'visa'],
        ['pcb2--108', 'visa'],
        ['pcb3--115', 'visa'],
        # ['pipe_fryum--070', 'visa'],

        # ['02-ko-057', 'btad'],
        ['01-ko-017', 'btad'],
        ['02-ko-079', 'btad'],
        ['03-ko-020', 'btad'],

        ['Blotchy_099-bad-011', 'dtd'],
        # ['Blotchy_099-bad-042', 'dtd'],
        ['Mesh_114-bad-076', 'dtd'],
        ['Perforated_037-bad-023', 'dtd'],

        ['class1-defect-071', 'button'],
        ['class2-defect-090', 'button'],
        # ['class2-defect-077', 'button'],
        ['class3-defect-040', 'button'],
        ['class4-defect-053', 'button'],

    ]

    save_name = 'main'

    save_dir = '/home/anyad/AnomalyDetection/test/main'

    plot_qualitative_results(root_dir=root_dir,
                             image_names=image_names,
                             method_dirs=method_dirs,
                             save_dir=save_dir,
                             save_name=save_name,
                             border_ratio = 0.01, gap_ratio = 0.1, image_size = 224, reverse=False)


    # ######## button
    # root_dir = '/home/anyad/AnomalyDetection/workspace'
    #
    # # varad
    # method_dirs = [
    #     ['PFM/images/{:}-1024', 'mamba'],  # mamba
    #     ['RD4AD/images/{:}-1024', 'mamba'],  # mamba
    #     ['PatchCore/images/{:}-wide_resnet50_2-1024', 'mamba'],  # mamba
    #     # ['UniAD/images/{:}-1024', 'UniAD'],
    #     ['CDO/images/{:}-hrnet32-1024', 'mamba'],  # mamba
    #     ['PyramidFlow/images/{:}-1024', 'mamba'],  # mamba
    #     ['AMI_Net/images/{:}-1024', 'mamba'],
    #     ['PNPT/images/{:}-1024', 'mamba'],
    #     ['VarAD/images/{:}-dinov2_vits14-1022-256-PROJ-True-[2, 3]--1', 'mamba'],  # mamba
    # ]
    #
    # image_names = [
    #     ['class1-defect-071', 'button'],
    #     ['class2-defect-077', 'button'],
    #     # ['class3-defect-040', 'button'],
    #     # ['class4-defect-053', 'button'],
    # ]
    #
    # save_name = 'button'
    #
    # save_dir = '/home/anyad/AnomalyDetection/test/button'
    #
    # plot_qualitative_results(root_dir=root_dir,
    #                          image_names=image_names,
    #                          method_dirs=method_dirs,
    #                          save_dir=save_dir,
    #                          save_name=save_name,
    #                          border_ratio = 0.01, gap_ratio = 0.1, image_size = 224, reverse=True)