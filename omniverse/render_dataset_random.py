# for isaac, isaac has different coordinate from code (-y,x)?
# bug: first two frames lose texture, use dummy frame
# random scene renderer, gt is center view
# https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html

import asyncio
import omni.replicator.core as rep
from datetime import datetime
import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom
import time

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

# setting
#  0 finished
run_time = 0  # todo, omniverse bug: frame gets slower, need restart; has it fixed?

have_position = 0  # 0-campute and save position.npy ;1-load position.npy

rep.set_global_seed(run_time)
np.random.seed(run_time)
out_path = 'D:/omniverse/train_dataset_random_new'  # save the rendered results
usds_path = 'D:/omniverse/collect_usds'  # usd model path
render_depth = 1  # render depth map of center view
run_scene = 10  # n scenes in a run
res = (3840,2160)
#res = (380,210)
spp = 64  # samples per pixel
rand_usd_num = 45  # number of usd
focus_num = 9  # 9
focus_round = 1  # f_stop
view_num = 7  # 6+2+1+1
f_stop1 = 0.94
dummy_frame = 2  # first render some frames, they will not be saved. 2

#f_list=[f_stop1,f_stop2,f_stop3,f_stop4,f_stop5,f_stop6]
f_list=[f_stop1]

center = 425  # z, 425
camera_dis = 8  # total_dis_range = 2 * dis
focus_dis = 6.25  # focus_range = focus_num * focus_dis,  50 / (focus_num-1)

# according to setting
rep.settings.set_render_pathtraced(samples_per_pixel=spp)
rep.settings.carb_settings("/rtx/pathtracing/maxBounces",32)  # default 4
rep.settings.carb_settings("/rtx/pathtracing/maxSpecularAndTransmissionBounces",32)  #default 6
rep.settings.carb_settings("/rtx/pathtracing/maxSamplesPerLaunch",100000000)

start_num = run_scene * run_time  # start index
view_num = view_num + dummy_frame
num_frame_per_scene = view_num + 1+focus_num * focus_round  # camera num,+1 random position all in focus
total_frame = run_scene * num_frame_per_scene
camera_position = [(-camera_dis, -camera_dis, center), (-camera_dis, camera_dis, center),
                   (camera_dis, -camera_dis, center),(camera_dis, camera_dis, center),
                   (0,-camera_dis, center), (0, camera_dis, center), (0, 0, center), (0, 0, center)]
camera_f_stop = [0, 0, 0, 0, 0, 0, 0,0]
camera_look_at = [(-camera_dis, -camera_dis, 0), (-camera_dis, camera_dis, 0),
                  (camera_dis, - camera_dis, 0), (camera_dis, camera_dis, 0),
                  (0,-camera_dis, 0), (0, camera_dis,  0), (0, 0, 0), (0, 0, 0)]
camera_focus_distance = [400, 400, 400, 400, 400, 400, 400,400]

# multifocal setting
for j in range(focus_num*focus_round):
    f_index = j % focus_num
    f_num = j/focus_num
    camera_position.append((0, 0, center))
    camera_f_stop.append(f_list[int(f_num)])
    camera_look_at.append((0, 0, 0))
    camera_focus_distance.append(int(400 + f_index * focus_dis))

if view_num == 1:  # debug mode: only render the center view
    camera_position.insert(0, (0, 0, center))
    camera_f_stop.insert(0, 0)
    camera_look_at.insert(0, (0, 0, 0))
    camera_focus_distance.insert(0, 400)

#dummy_frame setting
if dummy_frame>0:
    for j in range(dummy_frame):
        camera_position.insert(0,(0, 0, center))
        camera_f_stop.insert(0,0)
        camera_look_at.insert(0,(0, 0, 0))
        camera_focus_distance.insert(0,400)


if have_position==1:
    camera_position_random=np.load(out_path +'/camera_position_random.npy')
    print("using saved  position")
else:
    camera_position_random = []
    for j in range(1000):
        x = np.random.uniform(-camera_dis, camera_dis)
        y = np.random.uniform(-camera_dis, camera_dis)
        camera_position_random.append((x, y, center))
    np.save(out_path +'/camera_position_random.npy', camera_position_random)
    print("compute position and saved")

# create light, DistantLight
distant_lights = []
prim_type = "DistantLight"
next_free_path = omni.usd.get_stage_next_free_path(stage, f"/Replicator/{prim_type}", False)
light_prim = stage.DefinePrim(next_free_path, prim_type)
UsdGeom.Xformable(light_prim).AddTranslateOp().Set((0.0, 0.0, 0.0))
UsdGeom.Xformable(light_prim).AddRotateXYZOp().Set((0.0, 0.0, 0.0))
UsdGeom.Xformable(light_prim).AddScaleOp().Set((1.0, 1.0, 1.0))
light_prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1600.0)
distant_lights.append(light_prim)

# read usds and randomize
usds = rep.utils.get_usd_files(usds_path)
instances = rep.randomizer.instantiate(usds, size=rand_usd_num, mode='scene_instance')

# z coordinate
position_list=[]
for num in range(10000):
    posx=np.random.uniform(-35, 35)
    posy=np.random.uniform(-50, 50)
    front_behind = np.random.uniform(low=-1, high=1)
    if front_behind<-0.45:
    #if front_behind < 0.0:
        exp_beta = 0.7  #front (center+range)
        center = 0
        target_range = 14
        rand_z = np.random.rand()  # 0-1
        rand_z = np.exp(-rand_z * exp_beta)
        range_high = np.exp(0)
        range_low = np.exp(-exp_beta)
        rand_z=target_range + center + (range_low - rand_z) / (
                range_high - range_low) * target_range
        #rand_z = np.random.uniform(0, 14)
    else:
        exp_beta = 1.2 #behind (center-range)
        center = 0
        target_range = 23
        rand_z = np.random.rand()  # 0-1
        rand_z = np.exp(-rand_z * exp_beta)
        range_high = np.exp(0)
        range_low = np.exp(-exp_beta)
        rand_z = -target_range + center + (rand_z - range_low) / (
                range_high - range_low) * target_range
        #rand_z = np.random.uniform(-23, 0)
    #rand_z=np.random.uniform(-23, 14)
    position_list.append((posx, posy, rand_z))

def rand_scene():
    with instances:
        rep.modify.pose(
            position=rep.distribution.choice(position_list),
            #position=rep.distribution.sequence(position_list),
            rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
            scale=rep.distribution.uniform(1.8,2.0),
        )
    return instances.node

rep.randomizer.register(rand_scene)

# create camera
camera = rep.create.camera(position=camera_position[0], look_at=camera_look_at[0],
                               focus_distance=camera_focus_distance[0], f_stop=camera_f_stop[0],
                               focal_length=105,horizontal_aperture=36)  #fov~40  #105

# create render product
render_product = rep.create.render_product(camera, resolution=res)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(render_product)
if render_depth == 1:
    dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    dep.attach(render_product)
backend = rep.BackendDispatch({"paths": {"out_dir": out_path}})

async def run():
    time_start = time.time()
    print("run time ", run_time, " start")
    for frame_id in range(total_frame):

        scene_num = int(frame_id / num_frame_per_scene)
        scene_num = scene_num + start_num
        sub_camera_id = int(frame_id % num_frame_per_scene)

        #if sub_camera_id == 0:
        if sub_camera_id == 0:
            #if scene_num==run_time * run_scene:
            rep.randomizer.rand_scene()  # random scene
            for light in distant_lights:  # random light, rotate
                light.GetAttribute("inputs:intensity").Set(np.random.uniform(1300, 1700))
                light.GetAttribute("xformOp:rotateXYZ").Set(
                    (np.random.uniform(-30, 30), np.random.uniform(-30, 30), 0))

            print("render scene: ", scene_num,",  start time: ", datetime.now())

        # with camera:
        #     rep.modify.attribute(name='fStop', value=camera_f_stop[sub_camera_id])
        #     rep.modify.attribute(name='focusDistance',value=camera_focus_distance[sub_camera_id])
        #     rep.modify.pose(position=camera_position[sub_camera_id], look_at=camera_look_at[sub_camera_id])

        if sub_camera_id<view_num: #view 4+2+center
            with camera:
                rep.modify.attribute(name='fStop', value=camera_f_stop[sub_camera_id])
                rep.modify.attribute(name='focusDistance', value=camera_focus_distance[sub_camera_id])
                rep.modify.pose(position=camera_position[sub_camera_id], look_at=camera_look_at[sub_camera_id])

            out_name="scene"+str(scene_num) +"_view" + str(sub_camera_id-dummy_frame) + ".png"
        elif sub_camera_id==view_num:  # random viewpoint
            with camera:
                rep.modify.attribute(name='fStop', value=camera_f_stop[sub_camera_id])
                rep.modify.attribute(name='focusDistance', value=camera_focus_distance[sub_camera_id])
                random_pos = camera_position_random[scene_num]
                rep.modify.pose(position=random_pos, look_at=(random_pos[0], random_pos[1], 0))

            out_name="scene"+str(scene_num) +"_view" + str(sub_camera_id-dummy_frame) + ".png"
        else:  # multifocal
            with camera:
                rep.modify.attribute(name='fStop', value=camera_f_stop[sub_camera_id])
                rep.modify.attribute(name='focusDistance', value=camera_focus_distance[sub_camera_id])
                random_pos=camera_position_random[scene_num]
                rep.modify.pose(position=random_pos, look_at=(random_pos[0],random_pos[1],0))
            out_name = "scene" + str(scene_num) + "_f0" + "_focus" + str(
                (sub_camera_id - view_num - 1) % focus_num) + ".png"
            #out_name="scene"+str(scene_num) +"_f"+str(int((sub_camera_id-view_num)/focus_num)) +"_focus"+ str((sub_camera_id-view_num-1)%focus_num) + ".png"

        await rep.orchestrator.step_async(rt_subframes=spp)

        rgb_data = rgb.get_data()

        if sub_camera_id >= dummy_frame:
            backend.write_image(out_name, rgb_data)
            backend.wait_until_done()

        # save depth map for center view and random view
        if render_depth == 1:
            if sub_camera_id == (view_num - 1):
                dep_path = out_path + '/scene' + str(scene_num) + "_dep.npy"
                dep_data = dep.get_data()
                np.save(dep_path, dep_data)
            elif sub_camera_id == view_num:
                dep_path = out_path + '/scene' + str(scene_num) + "_dep_random.npy"
                dep_data = dep.get_data()
                np.save(dep_path, dep_data)

        if sub_camera_id == (num_frame_per_scene-1):
            time_end = time.time()
            time_sum = time_end - time_start
            print("scene ", scene_num, " finished,  time used(s): ",round(time_sum, 2))
            time_start = time.time()

    print("run time ", run_time, " finish, time:", datetime.now())
    rep.orchestrator.stop()

asyncio.ensure_future(run())