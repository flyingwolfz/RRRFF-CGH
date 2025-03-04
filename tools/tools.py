import numpy as np
import torch
import math
import cv2
from skimage.metrics import structural_similarity
import torch.nn as nn

# ----------------------------------------convert------------------------------------------

def lf2MacPI_color(lf, out_macpi):
    out_macpi[0::2, 0::2, :] = lf[0, 0, :, :, :]
    out_macpi[0::2, 1::2, :] = lf[0, 1, :, :, :]
    out_macpi[1::2, 0::2, :] = lf[1, 0, :, :, :]
    out_macpi[1::2, 1::2, :] = lf[1, 1, :, :, :]
    return out_macpi

def lf2MacPI_color_2view(lf, out_macpi):
    out_macpi[:, 0::2, :] = lf[0, :, :, :]
    out_macpi[:, 1::2, :] = lf[1, :, :, :]
    return out_macpi

def srgb_to_lin(image):
    thresh = 0.04045
    if torch.is_tensor(image):
        low_val = image <= thresh
        im_out = torch.zeros_like(image)
        im_out[low_val] = 25 / 323 * image[low_val]
        im_out[torch.logical_not(low_val)] = ((200 * image[torch.logical_not(low_val)] + 11)
                                              / 211) ** (12 / 5)
    else:
        im_out = np.where(image <= thresh, image / 12.92, ((image + 0.055) / 1.055) ** (12 / 5))
    return im_out

def lin_to_srgb(image):
    thresh = 0.0031308
    im_out = np.where(image <= thresh, 12.92 * image, 1.055 * (image ** (1 / 2.4)) - 0.055)
    return im_out

def amp_to_img(ampout,method='cut'):
    # 1,1,h,w amplitude to h,w image
    ampout = ampout.squeeze()
    ampout = ampout * ampout  # amp tp intensity
    ampout = ampout.to(torch.device("cpu")).data.numpy()
    ampout = lin_to_srgb(ampout)
    if method=='cut':
        ampout[ampout > 1] = 1
    elif method=='max':
        ampout = ampout / ampout.max()
    ampout =np.uint8(ampout * 255)
    return ampout

def amp_to_img_color(ampout,method='cut'):
    # 3,h,w amplitude to h,w image
    b=amp_to_img(ampout[0,:,:],method=method)
    g = amp_to_img(ampout[1, :, :],method=method)
    r = amp_to_img(ampout[2, :, :],method=method)
    color = cv2.merge([b, g, r])
    return color

def phase_to_img(holo_phase,mean=0,offset=0.0):
    # phase to image
    max_phs = 2 * torch.pi
    holo_phase = torch.squeeze(holo_phase)
    if mean==1:
        holo_phase = holo_phase - holo_phase.mean()
    #holo_phase = ((holo_phase + max_phs / 2) % max_phs) / max_phs
    holo_phase = ((holo_phase + offset/255.0*max_phs) % max_phs) / max_phs
    #holo_phase = ((holo_phase) % max_phs) / max_phs
    holo_phase = np.uint8(holo_phase.to(torch.device("cpu")).data.numpy() * 255)
    return holo_phase

def phase_to_img_color(holo_phase):
    # phase to image
    # b g r macpi
    holo0 = phase_to_img(holo_phase[:, 0, :, :], offset=30)
    holo1 = phase_to_img(holo_phase[:, 1, :, :], offset=130)
    holo2 = phase_to_img(holo_phase[:, 2, :, :], offset=30)

    color = cv2.merge([holo0, holo1, holo2])
    return color

# ---------------------------------------- psnr ssim ------------------------------------------

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_color(img1, img2):
    mse_0 = np.mean((cv2.split(img1)[0] - cv2.split(img2)[0]) ** 2)
    mse_1 = np.mean((cv2.split(img1)[1] - cv2.split(img2)[1]) ** 2)
    mse_2 = np.mean((cv2.split(img1)[2] - cv2.split(img2)[2]) ** 2)
    mse = (mse_0 + mse_1 + mse_2) / 3.0
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    return structural_similarity(img1, img2, data_range=1)

def ssim_color(img1, img2):
    ssim0 = structural_similarity(cv2.split(img1)[0], cv2.split(img2)[0], data_range=1)
    ssim1 = structural_similarity(cv2.split(img1)[1], cv2.split(img2)[1], data_range=1)
    ssim2 = structural_similarity(cv2.split(img1)[2], cv2.split(img2)[2], data_range=1)
    return (ssim0 + ssim1 + ssim2) / 3.0

# ----------------------------------------read image------------------------------------------


def load_img(path, color_channel=1, resize=0, res=(3840, 2160), to_amp=0,keep_numpy=0,device=torch.device("cuda:0")):
    #
    img = cv2.imread(path)
    if resize == 1:
        img = cv2.resize(img, res)
    if color_channel != 3:
        img = cv2.split(img)[color_channel]
    if keep_numpy == 0:  # 0-1 torch ,else 0-255 numpy
        img = torch.from_numpy(img).to(device)
        if color_channel != 3:
            img = img.unsqueeze(0).unsqueeze(0)  # x,y to 1,1,x,y
        else:
            img = img.permute(2, 0, 1).contiguous().unsqueeze(0)  # x,y,3 to 1,3,x,y
        img = img / 255.0
        if to_amp == 1:
            img = srgb_to_lin(img)
            img = torch.sqrt(img)
    return img

def get_dep_mask(dep=torch.zeros(1, 1, 1920, 1072), dep_interval=0.1, layer_num=1, shape=(1, 1, 1920, 1072),device=torch.device("cuda:0")):
    # mask using depth map
    mask = torch.zeros(shape[0], 1, shape[2], shape[3]).to(device)  # no need for color channel
    for k in range(layer_num):
        layer_index = k
        if layer_index == (layer_num - 1):
            mask[layer_index, :, :, :] = (dep >= (layer_index * dep_interval)) & (
                    dep <= (layer_index * dep_interval + dep_interval))
        else:
            mask[layer_index, :, :, :] = (dep >= (layer_index * dep_interval)) & (
                    dep < (layer_index * dep_interval + dep_interval))

    return mask

def get_input(data_path, image_index=0, input='macpi', color_channel=2,keep_numpy=0,device=torch.device("cuda:0")):
    if input == 'macpi':
        macpipath = data_path + 'scene_' + str(image_index) + '_macpi.png'
        net_input = load_img(path=macpipath, color_channel=color_channel, to_amp=1,keep_numpy=keep_numpy,device=device)
    elif input == 'macpi_2view':
        macpipath = data_path + 'scene_' + str(image_index) + '_macpi_2view.png'
        net_input = load_img(path=macpipath, color_channel=color_channel, to_amp=1,keep_numpy=keep_numpy,device=device)
    elif input == '1view':
        view_path = data_path + 'scene_' + str(image_index) + '_gt.png'
        net_input = load_img(path=view_path, color_channel=color_channel, to_amp=1,keep_numpy=keep_numpy,device=device)
    elif input == 'rgbd':
        view_path = data_path + 'scene_' + str(image_index) + '_gt.png'
        dep_path = data_path + 'scene_' + str(image_index) + '_dep.png'
        view = load_img(path=view_path, color_channel=color_channel, to_amp=1,keep_numpy=keep_numpy,device=device)
        dep = load_img(path=dep_path, color_channel=0, to_amp=0,keep_numpy=keep_numpy,device=device)
        net_input = torch.cat((view, dep), -3)
    return net_input


def get_gt(data_path=None, image_index=0, color_channel=2, shape=(1, 1, 1080, 1920),
           layer_num=1,resize=False,res=(3840, 2160),device=torch.device("cuda:0")):
    #todo, support for sgd
    gt = torch.zeros(shape).to(device)
    random_dep_path = data_path + 'scene_' + str(image_index) + '_randomdep.png'
    random_dep = load_img(path=random_dep_path, color_channel=0, resize=resize, res=res, to_amp=0, device=device)
    for i in range(layer_num):
        gtpath = data_path + 'scene_' + str(image_index) + '_f0_focus' + str(i) + '.png'
        gt[i, :, :, :] = load_img(path=gtpath, color_channel=color_channel, to_amp=1,resize=resize,res=res,device=device)

    return gt, random_dep


def get_rand_input(input_type='macpi', input_num=1, W=1920, H=1080, channel_num=1,device=torch.device("cuda:0")):
    if input_type == 'macpi':
        net_input = torch.randn(input_num, channel_num, H * 2, W * 2).to(device)
    elif input_type == 'macpi_2view':
        net_input = torch.randn(input_num, channel_num, H, W * 2).to(device)
    elif input_type == '1view':
        net_input = torch.randn(input_num, channel_num, H, W).to(device)
    elif input_type == 'rgbd':
        net_input = torch.randn(input_num, channel_num + 1, H, W).to(device)
    elif input_type == 'macpi_dual':
        net_input = torch.randn(input_num, channel_num*2, H * 2, W * 2).to(device)
    return net_input

def get_pad_size(input_type='macpi', cgh_H=1080,cgh_W=1920):
    if input_type == 'macpi':
        pad_size = (cgh_H*2,cgh_W*2)
    elif input_type == 'macpi_2view':
        pad_size = (cgh_H,cgh_W*2)
    elif input_type == '1view':
        pad_size = (cgh_H,cgh_W)
    elif input_type == 'rgbd':
        pad_size = (cgh_H,cgh_W)
    return pad_size
# ----------------------------------------ASM------------------------------------------
# from nh3d

def pad_stacked_complex(field, pad_width, padval=0, mode='constant'):
    """Helper for pad_image() that pads a real padval in a complex-aware manner"""
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width, mode=mode)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, mode=mode, value=padval)
        imag = nn.functional.pad(imag, pad_width, mode=mode, value=0)
        return torch.stack((real, imag), -1)


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,constant_values=padval)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field


def roll_torch(tensor, shift: int, axis: int):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def compute_H(input_field, prop_dist, wavelength, feature_size,
              F_aperture=0.5,device=torch.device("cuda:0")):

    num_y, num_x = input_field.shape[-2], input_field.shape[-1]  # number of pixels
    dx = feature_size  # sampling inteval size, pixel pitch of the SLM
    dy = feature_size

    # frequency coordinates sampling
    fy = torch.linspace(-1 / (2 * dy), 1 / (2 * dy), num_y).to(device)
    fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx), num_x).to(device)

    # momentum/reciprocal space
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    FX = torch.transpose(FX, 0, 1)
    FY = torch.transpose(FY, 0, 1)

    G = 2 * math.pi * (1 / wavelength ** 2 - (FX ** 2 + FY ** 2)).sqrt()
    H_exp = G.reshape((1, 1, *G.shape))

    fy_max = 1 / math.sqrt((2 * prop_dist * (1 / (dy * float(num_y)))) ** 2 + 1) / wavelength
    fx_max = 1 / math.sqrt((2 * prop_dist * (1 / (dx * float(num_x)))) ** 2 + 1) / wavelength
    H_filter = ((torch.abs(FX ** 2 + FY ** 2) <= (F_aperture ** 2) * torch.abs(FX ** 2 + FY ** 2).max())
                & (torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)).type(torch.FloatTensor)

    # if prop_dist==0:
    #     H = torch.ones_like(H_exp).to(device)
    # else:
    #     H = H_filter.to(device) * torch.exp(1j * H_exp * prop_dist)
    H = H_filter.to(device) * torch.exp(1j * H_exp * prop_dist)

    return H


def propagation_ASM(u_in, feature_size=0.0036, wavelength=0.000532, dis=1, linear_conv=True,
                    precomped_H=None, device=torch.device("cuda:0")):
    # if no H, calculate H, else ASM

    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        u_in = pad_image(u_in, conv_size, padval=0, stacked_complex=False)

    if precomped_H is None:
        H_total = []
        for this_dis in dis:
            H_color = []
            print('compute H for distance ', this_dis, 'mm with wavelength ', wavelength)
            for this_wavelength in wavelength:
                h = compute_H(input_field=torch.empty_like(u_in), prop_dist=this_dis,
                              wavelength=this_wavelength, feature_size=feature_size, F_aperture=0.5,device=device)
                H_color.append(h)
            H_color = torch.cat(H_color, dim=1)  # dis,color,h,w
            H_total.append(H_color)
        H = torch.cat(H_total, dim=0)  # dis,color,h,w
        print('H complete')
        return H
    else:
        U1 = torch.fft.fftshift(torch.fft.fftn(u_in, dim=(-2, -1), norm='ortho'), (-2, -1))
        U2 = U1 * precomped_H
        u_out = torch.fft.ifftn(torch.fft.ifftshift(U2, (-2, -1)), dim=(-2, -1), norm='ortho')
        if linear_conv:
            return crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)
        else:
            return u_out

def propagation_ASM_save(u_in, feature_size=0.0036, wavelength=0.000532, dis=1, linear_conv=True,
                    precomped_H=None, device=torch.device("cuda:0")):
    # if no H, calculate H, else ASM
    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        u_in = pad_image(u_in, conv_size, padval=0, stacked_complex=False)
    layer_num= precomped_H.size()[0]
    u_out=torch.zeros_like(precomped_H).to(device)
    for i in range(layer_num):
        U1 = torch.fft.fftshift(torch.fft.fftn(u_in, dim=(-2, -1), norm='ortho'), (-2, -1))
        U2 = U1 * precomped_H[i,:,:,:].to(device)
        u_out[i,:,:,:] = torch.fft.ifftn(torch.fft.ifftshift(U2, (-2, -1)), dim=(-2, -1), norm='ortho')
    return crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)

