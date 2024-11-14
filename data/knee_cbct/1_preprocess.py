import os
import json
import scipy
import numpy as np
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    return image


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


def load_names():
    with open('info.json', 'r') as f:
        info = json.load(f)
        names = []
        for s in ['train', 'test', 'eval']:
            names += info[s]
        return names

def convert_to_attenuation(data: np.array, rescale_slope: float, rescale_intercept: float):
    """
    CT scan is measured using Hounsfield units (HU). We need to convert it to attenuation.

    The HU is first computed with rescaling parameters:
        HU = slope * data + intercept

    Then HU is converted to attenuation:
        mu = mu_water + HU/1000x(mu_water-mu_air)
        mu_water = 0.206
        mu_air=0.0004

    Args:
    data (np.array(X, Y, Z)): CT data.
    rescale_slope (float): rescale slope.
    rescale_intercept (float): rescale intercept.

    Returns:
    mu (np.array(X, Y, Z)): attenuation map.

    """
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    # mu = mu * 100
    return mu

def resample():
    os.makedirs('./resampled', exist_ok=True)
    ref_spacing = np.array([0.8, 0.8, 0.8])
    nVoxels = np.array([256, 256, 256])

    for name in tqdm(load_names(), ncols=50):
        path = f'/home/foods/pro/NERF-test/data/{name}.mhd'

        itk_img = sitk.ReadImage(path)
        spacing = np.array(itk_img.GetSpacing())
        image = sitk.GetArrayFromImage(itk_img)
        spacing = (spacing[2], spacing[1], spacing[0])
        print('!!!!!', spacing, image.shape)

        image = image.transpose(2, 1, 0)
        imageDim = image.shape
        zoom_x = nVoxels[0] / imageDim[0]
        zoom_y = nVoxels[1] / imageDim[1]
        zoom_z = nVoxels[2] / imageDim[2]

        image = np.clip(image, a_min=-1000, a_max=3000)
        image = convert_to_attenuation(image, 1, 0)
        a_min = convert_to_attenuation(-1000, 1, 0)
        a_max = convert_to_attenuation(3000, 1, 0)
        image = (image - a_min) / (a_max-a_min)
        image = image.astype(np.float32)

        scaling = spacing / ref_spacing
        image = scipy.ndimage.zoom(
            image, 
            (zoom_x, zoom_y, zoom_z), 
            #scaling, 
            order=3, 
            prefilter=False
        )

        print('!!!!!', image.shape)
        save_path = f'./resampled/{name}.nii.gz'
        save_nifti(image, save_path)
        image = (image * 255).astype(np.uint8)


def crop_pad():
    os.makedirs('./processed', exist_ok=True)
    files = glob('./resampled/*.nii.gz')
    for file in tqdm(files, ncols=50):
        name = '.'.join(file.split('/')[-1].split('.')[:-2])
        image = read_nifti(file)
            
        # w, h
        if image.shape[0] > 256: # crop
            p = image.shape[0] // 2 - 128
            image = image[p:p+256, p:p+256, :]
        elif image.shape[0] < 256: # padding
            image_tmp = np.full([256, 256, image.shape[-1]], fill_value=0, dtype=np.uint8)
            p = 128 - image.shape[0] // 2
            l = image.shape[0]
            image_tmp[p:p+l, p:p+l, :] = image
            image = image_tmp

        # d
        if image.shape[-1] > 256: # crop
            p = image.shape[-1] // 2 - 128
            image = image[..., p:p+256]
        elif image.shape[-1] < 256: # padding
            image_tmp = np.full(list(image.shape[:2]) + [256], fill_value=0, dtype=np.uint8)
            p = 128 - image.shape[-1] // 2
            l = image.shape[-1]
            image_tmp[..., p:p+l] = image
            image = image_tmp

        save_path = f'./processed/{name}.nii.gz'
        save_nifti(image, save_path)

            
if __name__ == '__main__':
    resample()
    crop_pad()
        
