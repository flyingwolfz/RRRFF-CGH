# use rendered results, generate data for training
import torch
import cv2
import numpy
from tqdm import trange
import tools.tools as tools
import math
image_W=3840
image_H=2160

#position_out_path = 'D:/omniverse/valid_scene_glass2'
position_out_path = 'D:/omniverse/train_dataset_random_more'
camera_position_random=numpy.load(position_out_path +'/camera_position_random_cat.npy')
print("using saved  position")

W=3392
H=1728
# W=3744
# H=2080


# dataset
shiftx=422


#shiftx=15
#shiftx=72
shifty=shiftx
focus_num=0
datanum=310
start_index=0

#method='no'  # no macpi and view
method='4view'  # 0 1 2 3
#method='2view'   # 4 5

save_dep=1  # if have depth.npy
save_view=1  # save each view (after cut)
save_gt=1  # center view

dep_min=400-3.125
dep_max=450+3.125



lf_path = 'D:/omniverse/train_dataset_random_more/'
out_path = 'C:/zcl/python/train_dataset_random_more/'

gtcut1=int(image_H/2-H/2)
gtcut2=int(image_H/2+H/2)
gtcut3=int(image_W/2-W/2)
gtcut4=int(image_W/2+W/2)

if method=='4view':
    lf = torch.zeros(2, 2, H, W, 3)
    out_macpi = torch.zeros(H * 2, W * 2, 3)
elif method=='2view':
    lf = torch.zeros(2, H, W, 3)
    out_macpi = torch.zeros(H , W*2 , 3)

for k in trange(datanum):

    data_index=start_index+k
    camera_position=camera_position_random[data_index]
    gtshiftx = camera_position[1]
    gtshifty = camera_position[0]
    gtshiftx = math.floor(shiftx * (gtshiftx / (-8)) / 2)
    gtshifty = math.floor(shifty * (gtshifty / (-8)) / 2)
    shift_gtcut1 = int(gtcut1 + gtshifty)
    shift_gtcut2 = int(gtcut2 + gtshifty)
    shift_gtcut3 = int(gtcut3 + gtshiftx)
    shift_gtcut4 = int(gtcut4 + gtshiftx)

    if save_gt==1:

        gtpath = lf_path + 'scene' + str(data_index) + '_view6.png'
        gt = cv2.imread(gtpath)

        gt = gt[gtcut1:gtcut2, gtcut3:gtcut4, :]
        cv2.imwrite(out_path + 'scene_' + str(data_index) + '_gt.png', gt)

    if method=='4view':
        for k1 in range(2):
            for k2 in range(2):
                index = 2 * k1 + k2

                imgpath = lf_path + 'scene' + str(data_index) + '_view' + str(index) + '.png'
                img = cv2.imread(imgpath)
                img = torch.from_numpy(img)

                cut1 = int(gtcut1 + shifty / 2 - k1 * shifty)
                cut2 = int(gtcut2 + shifty / 2 - k1 * shifty)
                cut3 = int(gtcut3 + shiftx / 2 - k2 * shiftx)
                cut4 = int(gtcut4 + shiftx / 2 - k2 * shiftx)

                img = img[cut1:cut2, cut3:cut4, :]

                lf[k1, k2, :, :, :] = img

                if save_view == 1:
                    img = img.numpy()
                    img = numpy.uint8(img)
                    view_path = out_path + 'scene_' + str(data_index) + '_view' + str(index) + '.png'
                    cv2.imwrite(view_path, img)

        macpi = tools.lf2MacPI_color(lf, out_macpi)
        macpi = macpi.numpy()
        macpi = numpy.uint8(macpi)
        cv2.imwrite(out_path + 'scene_' + str(data_index) + '_macpi.png', macpi)

    elif method=='2view':
        for k1 in range(2):
            index = 4 + k1
            imgpath = lf_path + 'scene' + str(data_index) + '_view' + str(index) + '.png'
            img = cv2.imread(imgpath)
            img = torch.from_numpy(img)

            cut1 = int(gtcut1)
            cut2 = int(gtcut2)
            cut3 = int(gtcut3 + shiftx / 2 - k1 * shiftx)
            cut4 = int(gtcut4 + shiftx / 2 - k1 * shiftx)

            img = img[cut1:cut2, cut3:cut4, :]

            lf[k1, :, :, :] = img

            if save_view == 1:
                img = img.numpy()
                img = numpy.uint8(img)
                view_path = out_path + 'scene_' + str(data_index) + '_view' + str(index) + '.png'
                cv2.imwrite(view_path, img)

        macpi = tools.lf2MacPI_color_2view(lf, out_macpi)
        macpi = macpi.numpy()
        macpi = numpy.uint8(macpi)
        cv2.imwrite(out_path + 'scene_' + str(data_index) + '_macpi_2view.png', macpi)

    if save_dep == 1:
        #dep_path = lf_path + 'scene' + str(data_index) + '_dep.npy'
        dep_path = lf_path + 'scene' + str(data_index) + '_dep_random.npy'
        dep = numpy.load(dep_path)
        dep[dep > dep_max] = dep_max
        dep[dep < dep_min] = dep_min
        dep = (dep - dep_min) / (dep_max - dep_min)
        dep = numpy.uint8(dep * 255)
        #dep = dep[gtcut1:gtcut2, gtcut3:gtcut4]
        dep = dep[shift_gtcut1:shift_gtcut2, shift_gtcut3:shift_gtcut4]
        cv2.imwrite(out_path + 'scene_' + str(data_index) + '_randomdep.png', dep)

        dep_path = lf_path + 'scene' + str(data_index) + '_dep.npy'
        dep = numpy.load(dep_path)
        dep[dep > dep_max] = dep_max
        dep[dep < dep_min] = dep_min
        dep = (dep - dep_min) / (dep_max - dep_min)
        dep = numpy.uint8(dep * 255)
        dep = dep[gtcut1:gtcut2, gtcut3:gtcut4]
        cv2.imwrite(out_path + 'scene_' + str(data_index) + '_dep.png', dep)

    for k3 in range(focus_num):
        focuspath = lf_path+ 'scene' + str(data_index) + '_f0_focus'+str(k3)+'.png'
        focus = cv2.imread(focuspath)
        focus = focus[shift_gtcut1:shift_gtcut2, shift_gtcut3:shift_gtcut4, :]
        focus_outpath =  out_path+'scene_'+ str(data_index)+ '_f0_focus'+str(k3)+'.png'
        cv2.imwrite(focus_outpath , focus)