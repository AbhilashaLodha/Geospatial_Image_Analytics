import random
import numpy as np
import logging
import json
from keras import backend as K

logger = logging.getLogger('Gen Patches........')
logger.setLevel(logging.DEBUG)


def get_rand_patch(img, mask, sz=160):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    logger.info("sz---%s"%sz)
    logger.info("shape1--%s"%img.shape[0])
    logger.info("shape2--%s"%img.shape[1])
    logger.info("shape3--%s"%img.shape[2])
    logger.info("shape4--%s"%mask.shape[0])
    logger.info("shape5--%s"%mask.shape[1])
    logger.info("shape6--%s"%mask.shape[2])
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    #assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz 
    logger.info("line no 20")
    print("____________-------")
    xc = random.randint(0, img.shape[0] - sz)
    logger.info("line no 21")
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]
    logger.info("line no 25")

    # Apply some random transformations
    random_transformation = np.random.randint(1,8)
    logger.info("line no 29")
    if random_transformation == 1:  # reverse first dimension
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    else:
        pass

    return patch_img, patch_mask


def get_patches(x_dict, y_dict, n_patches, sz=160):
    x = list()
    y = list()
    total_patches = 0
    logger.info("got get_patches")
    logger.info("n patches---%s"%n_patches)
    #logger.info("got get_patches")
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        logger.info("line no 62")
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        logger.info("line no 64")
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)
