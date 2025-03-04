import model
import tools.tools as tools
import torch
import numpy
import cv2
from tqdm import trange

load_path = 'train_dataset_random_more/'
#load_path = 'valid_scene_glass/'
#load_path = 'valid_scene_dartboard/'

load_index_start = 300  # start
load_num = 10

# load_index_start = 0  # start
# load_num = 9

camera_position_random=numpy.load(load_path+'camera_position_random_cat.npy')
#camera_position_random=numpy.load(load_path+'camera_position.npy')
print("using saved  position")

have_gt = 1 # using focus supervision,  get psnr #and ssim

train_sup = 'focus'
add_sup = 'focus'


input_type = 'macpi'
#input_type='1view'   #rgb
#input_type='rgbd'
#input_type='macpi_2view'


# for img
img_W=3392
img_H=1728

# for CGH
# cgh_W=3840
# cgh_H=2160
cgh_W=3392
cgh_H=1728

pad_res=tools.get_pad_size(input_type=input_type, cgh_H=cgh_H,cgh_W=cgh_W)

pitch = 0.0036
color_channel = 3  # 0,1,2 for b,g,r ; 3 for full color
wavelength = [0.000457, 0.000532, 0.000639]
layer_num = 9  # at least 1, depth_range=(layernum-1)*interval
interval = 0.625 # between layers
z = -5  # nearest

device_cuda0 = torch.device("cuda:0")  # for network and asm
device_cuda1 = torch.device("cuda:0")  # for  loss
device_cpu=torch.device("cpu")

if color_channel == 3:  # only 3 is support here
    channel_num = 3
else:
    channel_num = 1
    wavelength = [wavelength[color_channel]]  # use list here
dis = []
for k in range(layer_num):
    dis.append(z + k * interval)  # list

img_shape = (layer_num, channel_num, img_H, img_W)
cgh_shape = (layer_num, channel_num, cgh_H, cgh_W)

Hbackward = tools.propagation_ASM(u_in=torch.empty(1, 1, cgh_H, cgh_W), feature_size=pitch, dis=dis,
                                  wavelength=wavelength, precomped_H=None,device=device_cuda0)  # return H
Hbackward=Hbackward.requires_grad_(requires_grad=False)

net = model.Net_random(input_type=input_type, channel_num=channel_num).to(device_cuda0)
net=net.requires_grad_(requires_grad=False)

model_name='input_' + input_type + '_sup_' + train_sup + '_addsup_' + str(add_sup) + '_channel_' + str(color_channel)
net.load_state_dict(torch.load('trained_model/'+model_name + '.pth'))

averagepsnr=0
averagessim=0


for i in trange(load_num):
    psnr_value = 0
    ssim_value = 0
    load_index=load_index_start+i

    camera_position = camera_position_random[load_index]
    camera_positiony = camera_position[0]
    camera_positionx = camera_position[1]
    posx = torch.full((1, 1, img_H, img_W), camera_positionx / 8.0).cuda(0)
    posy = torch.full((1, 1, img_H, img_W), camera_positiony / 8.0).cuda(0)

    net_input = tools.get_input(data_path=load_path, image_index=load_index,
                                input=input_type, color_channel=color_channel, device=device_cuda0)

    if img_shape != cgh_shape:
        net_input = tools.pad_image(net_input, pad_res, pytorch=True, stacked_complex=False)

    with torch.no_grad():
        holo_phase = net(net_input,posx,posy)
        slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
        slm_complex = slm_complex.to(device_cuda0)
        for index in range(layer_num):
            recon_complex = tools.propagation_ASM(u_in=slm_complex, precomped_H=Hbackward[index, :, :, :])
            ampout = torch.abs(recon_complex)

            if img_shape != cgh_shape:
                ampout = tools.crop_image(ampout, (img_H, img_W), pytorch=True, stacked_complex=False)
            if color_channel == 3:
                img_out = tools.amp_to_img_color(torch.squeeze(ampout),method='cut')  # C,H,W
                if have_gt == 1:
                    gt_path = load_path + 'scene_' + str(load_index) + '_f0_focus' + str(index) + '.png'
                    gt = cv2.imread(gt_path)

                    psnr_value+=tools.psnr_color(img_out / 255.0, gt / 255.0)
                    ssim_value+=tools.ssim_color(img_out / 255.0, gt / 255.0)

                    img1 = torch.Tensor(img_out/ 255.0).permute(2, 0, 1).unsqueeze(0).to(device_cuda1)
                    img2 = torch.Tensor(gt/ 255.0).permute(2, 0, 1).unsqueeze(0).to(device_cuda1)


    averagepsnr += psnr_value / layer_num
    averagessim += ssim_value / layer_num


if have_gt == 1:
    print('averagepsnr:', round(averagepsnr / load_num, 4))
    print('averagessim:', round(averagessim / load_num, 4),)
