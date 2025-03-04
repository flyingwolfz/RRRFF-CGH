
# 2 4090D GPU cards are used (2*24GB GPU memory)

import model
from tqdm import trange
import tools.tools as tools
import torch
import time
import numpy

seed = 12  # same seed for all methods, but different pattern need different  offsets
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

camera_position_random=numpy.load('train_dataset_random_more/camera_position_random_cat.npy')
print("using saved  position")

loss_save_index = 4

data_path = 'train_dataset_random_more/'
out_path = 'out/'

# for img
img_W=3392
img_H=1728

# for CGH
# cgh_W=3840
# cgh_H=2160

cgh_W=3392
cgh_H=1728

# supervision for train
train_sup = 'focus'

# additional sup for in focus area
add_sup = 'focus'
add_weight = 10

#input_type = 'macpi'
input_type='1view'
#input_type='rgbd'
#input_type='macpi_2view'


input_type_net=input_type

pad_res=tools.get_pad_size(input_type=input_type, cgh_H=cgh_H,cgh_W=cgh_W)

use_buffer = 0  # if you have big memory

pitch = 0.0036
color_channel = 3  # 0,1,2 for b,g,r ; 3 for full color
wavelength = [0.000457, 0.000532, 0.000639]

epoch_num =100
data_num = 300 # for train
lr = 0.0005

layer_num = 9  # at least 1, depth_range=(layernum-1)*interval
interval = 0.625  # between layers
z = -5.0

device_cuda0 = torch.device("cuda:0")  # for network
device_cuda1 = torch.device("cuda:1")  # for asm and loss
device_cpu = torch.device("cpu")

if color_channel == 3:
    channel_num = 3
else:
    channel_num = 1
    wavelength = [wavelength[color_channel]]  # use list here

img_shape = (layer_num, channel_num, img_H, img_W)
cgh_shape = (layer_num, channel_num, cgh_H, cgh_W)

dis = []
for k in range(layer_num):
    dis.append(z + k * interval)  # list

print('running index:', loss_save_index, ' input:', input_type, ' supervision:', train_sup, 'channel:', color_channel)
pth_name = 'trained_model/input_' + input_type + '_sup_' + train_sup + '_addsup_'+str(add_sup)+'_channel_' + str(color_channel) + '.pth'
print('pth will save as:', pth_name)

input_buffer, gt_dep_buffer,  mask, loss, gt_dep = None, None, None, None, None
if use_buffer == 1:
    print("load buffer......")
    input_buffer = tools.get_rand_input(input_type=input_type, input_num=data_num, W=img_W, H=img_H,
                                        channel_num=channel_num, device=device_cpu)
    gt_buffer = torch.zeros(data_num * layer_num, channel_num, img_H,img_W).to(device_cpu)
    if add_sup == 'focus':
        gt_dep_buffer = torch.zeros(data_num, 1, img_H, img_W).to(device_cpu)

    for i in trange(data_num):
        image_index = i
        input_buffer[image_index, :, :, :] = tools.get_input(data_path=data_path, image_index=image_index,
                                                             input=input_type,
                                                             color_channel=color_channel, device=device_cpu)
        gt,  gt_dep = tools.get_gt(data_path=data_path, image_index=image_index,
                                          shape=img_shape, color_channel=color_channel, layer_num=layer_num,
                                          device=device_cpu)
        gt_buffer[image_index * layer_num:(image_index + 1) * layer_num, :, :, :] = gt
        if add_sup == 'focus':
            gt_dep_buffer[image_index, :, :, :] = gt_dep

Hbackward = tools.propagation_ASM(u_in=torch.empty(1, 1, cgh_H, cgh_W), feature_size=pitch, dis=dis,
                                  wavelength=wavelength, precomped_H=None, device=device_cuda1)  # return H

net = model.Net_random(input_type=input_type_net, channel_num=channel_num).to(device_cuda0)
optvars = [{'params': net.parameters()}]
optimizer = torch.optim.Adam(optvars, lr=lr)
criterion_Loss = torch.nn.MSELoss().to(device_cuda1)

train_loss = []
test_loss = []

print("training......")
time.sleep(0.1)

for k in trange(epoch_num):
    currentloss = 0
    for kk in range(data_num):
        image_index = kk

        camera_position = camera_position_random[image_index]
        camera_positiony = camera_position[0]
        camera_positionx = camera_position[1]
        posx = torch.full((1,1,img_H, img_W), camera_positionx/8.0).cuda(0)
        posy = torch.full((1, 1, img_H, img_W), camera_positiony/8.0).cuda(0)

        if use_buffer == 1:
            net_input = input_buffer[image_index, :, :, :].unsqueeze(0).to(device_cuda0)
            gt = gt_buffer[image_index * layer_num:(image_index + 1) * layer_num, :, :, :].to(device_cuda1)
            gt_dep = gt_dep_buffer[image_index, :, :, :].to(device_cuda1)

        else:
            net_input = tools.get_input(data_path=data_path, image_index=image_index,
                                        input=input_type, color_channel=color_channel, device=device_cuda0)
            gt, gt_dep = tools.get_gt(data_path=data_path,
                                              image_index=image_index,
                                              shape=img_shape, color_channel=color_channel, layer_num=layer_num,
                                              device=device_cuda1)

        if img_shape != cgh_shape:
            net_input = tools.pad_image(net_input, pad_res, pytorch=True, stacked_complex=False)

        holo_phase = net(net_input,posx,posy)
        slm_complex = torch.complex(torch.cos(holo_phase), torch.sin(holo_phase))
        slm_complex = slm_complex.to(device_cuda1)

        for i in range(layer_num):
            recon_complex = tools.propagation_ASM(u_in=slm_complex, precomped_H=Hbackward[i, :, :, :],
                                                  device=device_cuda1)
            ampout = torch.abs(recon_complex)
            if img_shape != cgh_shape:
                ampout = tools.crop_image(ampout, (img_H, img_W), pytorch=True, stacked_complex=False)

            if train_sup == 'focus':
                loss = criterion_Loss(ampout, gt[i, :, :, :].unsqueeze(0))
            if add_sup == 'focus':
                this_dep = i / 8.0  # i 0-8
                weight2 = torch.exp(-40.0 * (gt_dep - this_dep) * (gt_dep - this_dep))
                loss2 = criterion_Loss(ampout *weight2,gt[i, :, :, :].unsqueeze(0)*weight2)
                loss += add_weight * loss2

            loss.backward(retain_graph=True)
            currentloss = currentloss + loss.cpu().data.numpy()/ layer_num

        optimizer.step()
        optimizer.zero_grad()

    train_loss.append(currentloss / data_num)
    if (k + 1) % 1 == 0 or (k + 1) == epoch_num:
        # no test to save memory
        print('trainloss:', currentloss / data_num)
        time.sleep(0.1)

numpy.save('loss/' + str(loss_save_index) + 'train_loss.npy', train_loss)
torch.save(net.state_dict(), pth_name)