from typing import Callable
import os
import os.path as osp

import pickle
import datetime

from rozumarm_vima_utils.scene_renderer import VIMASceneRenderer
from rozumarm_vima_utils.robot import RozumArm
from rozumarm_vima_utils.camera import Camera, CamDenseReader
from rozumarm_vima.rozumarm_vima_cv.segment_scene import segment_scene

import numpy as np
import cv2
from time import sleep

from rozumarm_vima_utils.transform import rf_tf_c2r, map_tf_repr_c2r
import argparse


USE_OBS_FROM_SIM = False
USE_ORACLE = False

N_SWEPT_OBJECTS = 2
USE_REAL_ROBOT = True
# USE_FIXED_PROMPT_FOR_SIM = False

WRITE_TRAJS_TO_DATASET = False
DATASET_DIR = "rozumarm-dataset"


def prepare_sim_obs(detector, env_renderer):
    n_cubes = -1
    while n_cubes != 2 * N_SWEPT_OBJECTS:
        obj_posquats = detector.detect()
        n_cubes = len(obj_posquats)

    # map from cam to rozum
    obj_posquats = [
        (rf_tf_c2r(pos), map_tf_repr_c2r(quat)) for pos, quat in obj_posquats
    ]

    env_renderer.render_scene(obj_posquats)
    obs, *_ = env_renderer.env.step(action=None)

    _, top_cam_image = detector.cam_1.read_image()
    _, front_cam_image = detector.cam_2.read_image()
    return obs, top_cam_image, front_cam_image


def prepare_real_obs(top_cam, front_cam):
    _, image_top = top_cam.read_image()
    _, image_front = front_cam.read_image()

    segm_top, _ = segment_scene(image_top, "top")
    segm_front, _ = segment_scene(image_front, "front")

    img_top = cv2.resize(image_top, (256, 128))
    img_front = cv2.resize(image_front, (256, 128))

    img_top = cv2.rotate(img_top, cv2.ROTATE_180)
    segm_top = cv2.rotate(segm_top, cv2.ROTATE_180)

    obs = {
        'rgb': {
            'front': np.transpose(img_front, axes=(2, 0, 1)),
            'top': np.transpose(img_top, axes=(2, 0, 1))
        },
        'segm':{
            'front': segm_front,
            'top': segm_top
        },
        'ee': 1  # spatula
    }
    return obs


def run_loop(r, robot, oracle, model=None, n_iters=1):
    """
    r: scene renderer
    """
    if WRITE_TRAJS_TO_DATASET:
        os.makedirs(DATASET_DIR, exist_ok=True)
        traj = []

    if USE_OBS_FROM_SIM:
        from rozumarm_vima.detectors import detector
        cubes_detector = detector

        prompt_assets = r.env.prompt_assets
    else:
        cam_1 = CamDenseReader(0, 'cam_top_video.mp4')
        cam_2 = CamDenseReader(2, 'cam_front_video.mp4')
        cam_1.start_recording()
        cam_2.start_recording()

        prompt_assets = get_prompt_assets()

    if not USE_ORACLE:
        # prompt = r.env.prompt
        prompt = 'Sweep all /{swept_obj/} into /{bounds/} without exceeding /{constraint/}'
        model.reset(prompt, prompt_assets)

    while True:
        for i in range(n_iters):
            if USE_OBS_FROM_SIM:
                obs, top_cam_image, front_cam_image = prepare_sim_obs(cubes_detector, r)
            else:
                obs = prepare_real_obs(cam_1, cam_2)

            if USE_ORACLE:
                action = oracle.act(obs)
                # print(f'Prompt is {r.env.prompt}')
                # print(f'action is {action}')
                # print('---------------------------')
            else:
                meta_info = {'action_bounds':{'low': np.array([ 0.25, -0.5 ]), 'high': np.array([0.75, 0.5 ])}}
                meta_info["n_objects"] = 4
                meta_info["obj_id_to_info"] = {4: {'obj_name': 'three-sided rectangle'},
                                                5: {'obj_name': 'line'},
                                                6: {'obj_name': 'small block'},
                                                7: {'obj_name': 'small block'}}
                action = model.step(obs, meta_info)
                # with open(r'student__{}.dat'.format(model.elapsed_steps),'w+b') as file: 
                #      obs['action'] = action
                #      obs['prompt'] = prompt
                #      obs['assets'] = r.env.prompt_assets
                #      #obs['inf_cache'] = model.inference_cache
                #      pickle.dump(obs,file)

            if action is None:
                print("Press Enter to try again, or q + Enter to exit.")
                ret = input()
                if len(ret) > 0 and ret[0] == 'q':
                    return
                r.reset(exact_num_swept_objects=N_SWEPT_OBJECTS)
                continue
                # print("ORACLE FAILED.")
                # # cubes_detector.release()
                # return

            # print(action) #.detach().cpu().numpy()[0][0]

            clipped_action = {
                k: np.clip(v, r.env.action_space[k].low, r.env.action_space[k].high)
                for k, v in action.items()
            }

            pos_0 = clipped_action["pose0_position"]
            pos_1 = clipped_action["pose1_position"]
            eef_quat = robot.get_swipe_quat(pos_0, pos_1)
            
            # x_compensation_bias = 0.03
            x_compensation_bias = 0.0
            pos_0[0] += x_compensation_bias
            pos_1[0] += x_compensation_bias
            
            posquat_0 = (pos_0, eef_quat)
            posquat_1 = (pos_1, eef_quat)
            robot.swipe(posquat_0, posquat_1)
            r.env.step(action)

            if WRITE_TRAJS_TO_DATASET:
                if USE_OBS_FROM_SIM:
                    obs["top_cam_image"] = top_cam_image
                    obs["front_cam_image"] = front_cam_image
                obs["after_sim_swipe"] = r.env._get_obs()
                traj.append(obs)
                traj.append(clipped_action)

        print("Press Enter to try again, n to New Episode; or q + Enter to exit.")
        ret = input()
        if ret == "s":
            assert WRITE_TRAJS_TO_DATASET, "Cannot write trajectory."
            if USE_OBS_FROM_SIM:
                obs, top_cam_image, front_cam_image = prepare_sim_obs(cubes_detector, r)
                obs["top_cam_image"] = top_cam_image
                obs["front_cam_image"] = front_cam_image
            else:
                obs = prepare_real_obs(cam_1, cam_2)
            traj.append(obs)

            file_id = datetime.datetime.now().isoformat()
            filepath = f"{DATASET_DIR}/{file_id}.traj"
            with open(filepath, 'wb') as f:
                pickle.dump(traj, f)
            print(f"Saved trajectory {file_id}.")
            traj.clear()

            ret = input("Enter command: ")
        if len(ret) > 0 and ret[0] == 'q':
            if USE_OBS_FROM_SIM:
                detector.release()
            return
        if len(ret) > 0 and ret[0] == 'n':
            r.reset(exact_num_swept_objects=N_SWEPT_OBJECTS)

            if WRITE_TRAJS_TO_DATASET:
                traj.clear()

            if not USE_ORACLE:
                if USE_OBS_FROM_SIM:
                    prompt_assets = r.env.prompt_assets
                # prompt = r.env.prompt
                prompt = 'Sweep all /{swept_obj/} into /{bounds/} without exceeding /{constraint/}'
                model.reset(prompt, prompt_assets)
        continue
    
    if USE_OBS_FROM_SIM:
        detector.release()


# def run_loop_sim_to_real(r, prompt_assets, robot, oracle, model=None, n_iters=3):
#     """
#     r: scene renderer
#     """

#     cam_1 = CamDenseReader(0, 'cam_top_video.mp4')
#     cam_2 = CamDenseReader(4, 'cam_front_video.mp4')
#     cam_1.start_recording()
#     cam_2.start_recording()
#     sleep(3)

#     counter = 1
#     while True:
#         counter += 1
#         for i in range(n_iters):
#             _, image_top = cam_1.read_image()
#             _, image_front = cam_2.read_image()

#             segm_top, _ = segment_scene(image_top, "top")
#             segm_front, _ = segment_scene(image_front, "front")

#             img_top = cv2.resize(image_top, (256, 128))
#             img_front = cv2.resize(image_front, (256, 128))

#             img_top = cv2.rotate(img_top, cv2.ROTATE_180)
#             segm_top = cv2.rotate(segm_top, cv2.ROTATE_180)

#             obs = {
#                 'rgb': {
#                     'front': np.transpose(img_front, axes=(2, 0, 1)),
#                     'top': np.transpose(img_top, axes=(2, 0, 1))
#                 },
#                 'segm':{
#                     'front': segm_front,
#                     'top': segm_top
#                 },
#                 'ee': 1  # spatula
#             }

#             with open(f"model_input_{counter}.pickle", 'wb') as f:
#                 pickle.dump({'obs': obs, 'prompt_assets': prompt_assets}, f, protocol=pickle.HIGHEST_PROTOCOL)

#             meta_info = {'action_bounds':{'low': np.array([ 0.25, -0.5 ]), 'high': np.array([0.75, 0.5 ])}}
            
#             meta_info["n_objects"] = 4
#             meta_info["obj_id_to_info"] = {4: {'obj_name': 'three-sided rectangle'},
#                                             5: {'obj_name': 'line'},
#                                             6: {'obj_name': 'small block'},
#                                             7: {'obj_name': 'small block'}}
#             model.reset(r.env.prompt, prompt_assets)

#             # action = model.step(obs,meta_info)
#             action = oracle.act(obs)

#             if action is None:
#                 print("Press Enter to try again, or q + Enter to exit.")
#                 ret = input()
#                 if len(ret) > 0 and ret[0] == 'q':
#                     cam_1.stop_recording()
#                     cam_2.stop_recording()
#                     return
#                 r.reset(exact_num_swept_objects=1)
#                 continue
#                 # print("ORACLE FAILED.")
#                 # # cubes_detector.release()
#                 # return

#             clipped_action = {
#                 k: np.clip(v, r.env.action_space[k].low, r.env.action_space[k].high)
#                 for k, v in action.items()
#             }

#             pos_0 = clipped_action["pose0_position"]
#             pos_1 = clipped_action["pose1_position"]
#             eef_quat = robot.get_swipe_quat(pos_0, pos_1)
            
#             x_compensation_bias = 0.03
#             pos_0[0] += x_compensation_bias
#             pos_1[0] += x_compensation_bias
            
#             posquat_0 = (pos_0, eef_quat)
#             posquat_1 = (pos_1, eef_quat)
#             robot.swipe(posquat_0, posquat_1)

#         print("Press Enter to start over...")
#         ret = input()
#         if len(ret) > 0 and ret[0] == 'q':
#             cam_1.stop_recording()
#             cam_2.stop_recording()
#             return
#         r.reset(exact_num_swept_objects=1)
#         model.reset(r.env.prompt,r.env.prompt_assets)
#         continue


def get_prompt_assets():
    folder = "/home/daniil/code/rozumarm-vima/rozumarm_vima/rozumarm_vima_cv/images/prompts/"

    bounds = dict()
    bounds['rgb'] = dict()
    bounds['rgb']['top'] = cv2.imread(folder + "img/goal_top.png").transpose(2, 0, 1)
    bounds['rgb']['front'] = cv2.imread(folder + "img/goal_front.png").transpose(2, 0, 1)
    bounds['segm'] = dict()
    bounds['segm']['top'] = cv2.imread(folder + "segm/goal_top.png", cv2.IMREAD_GRAYSCALE)
    bounds['segm']['front'] = cv2.imread(folder + "segm/goal_front.png", cv2.IMREAD_GRAYSCALE)
    bounds['segm']['obj_info'] = {
        'obj_id': 0,
        'obj_name': 'three-sided rectangle',
        'obj_color': 'red and blue stripe'}
    bounds['placeholder_type'] = 'object'

    constraint = dict()
    constraint['rgb'] = dict()
    constraint['rgb']['top'] = cv2.imread(folder + "img/stop_line_top.png").transpose(2, 0, 1)
    constraint['rgb']['front'] = cv2.imread(folder + "img/stop_line_front.png").transpose(2, 0, 1)
    constraint['segm'] = dict()
    constraint['segm']['top'] = cv2.imread(folder + "segm/stop_line_top.png", cv2.IMREAD_GRAYSCALE)
    constraint['segm']['front'] = cv2.imread(folder + "segm/stop_line_front.png", cv2.IMREAD_GRAYSCALE)
    constraint['segm']['obj_info'] = {
        'obj_id': 0,
        'obj_name': 'line',
        'obj_color': 'yellow and blue stripe'}
    constraint['placeholder_type'] = 'object'

    box_name = "red_box"
    swept_obj = dict()
    swept_obj['rgb'] = dict()
    swept_obj['rgb']['top'] = cv2.imread(folder + f"img/{box_name}_top.png").transpose(2, 0, 1)
    swept_obj['rgb']['front'] = cv2.imread(folder + f"img/{box_name}_front.png").transpose(2, 0, 1)
    swept_obj['segm'] = dict()
    swept_obj['segm']['top'] = cv2.imread(folder + f"segm/{box_name}_top.png", cv2.IMREAD_GRAYSCALE)
    swept_obj['segm']['front'] = cv2.imread(folder + f"segm/{box_name}_front.png", cv2.IMREAD_GRAYSCALE)
    swept_obj['segm']['obj_info'] = {
        'obj_id': 0,
        'obj_name': 'small block',
        'obj_color': 'yellow and blue polka dot'}
    swept_obj['placeholder_type'] = 'object'

    prompt_assets = dict()
    prompt_assets['bounds'] = bounds
    prompt_assets['constraint'] = constraint
    prompt_assets['swept_obj'] = swept_obj

    return prompt_assets


def main():
    r = VIMASceneRenderer('sweep_without_exceeding')
    r.reset(exact_num_swept_objects=N_SWEPT_OBJECTS)

    robot = RozumArm(use_mock_api=not USE_REAL_ROBOT)

    arg = argparse.ArgumentParser()
    arg.add_argument("--partition", type=str, default="placement_generalization")
    arg.add_argument("--task", type=str, default="visual_manipulation")
    arg.add_argument("--ckpt", type=str, required=True)
    arg.add_argument("--device", default="cpu")
    arg = arg.parse_args("--ckpt ./20M.ckpt --device cuda --task sweep_without_exceeding".split())    
    
    if USE_ORACLE:
        oracle = r.env.task.oracle(r.env)
        model = None
    else:
        oracle = None
        from rozumarm_vima.vima_model import VimaModel
        from rozumarm_vima.rudolph_model import RuDolphModel
        # model = VimaModel(arg)
        model = RuDolphModel()

    run_loop(r, robot, oracle, model=model, n_iters=1)


"""
from rozumarm_vima_utils.rozum_env import RozumEnv
def main_from_env():
    rozum_env = RozumEnv()

    obs = rozum_env.reset()
    for i in range(5):
        oracle = rozum_env.renderer.env.task.oracle(rozum_env.renderer.env)
        action = oracle.act(obs)
        if action is None:
            print("ORACLE FAILED.")
            return

        clipped_action = {
            k: np.clip(v, rozum_env.renderer.env.action_space[k].low, rozum_env.renderer.env.action_space[k].high)
            for k, v in action.items()
        }
        obs = rozum_env.step(clipped_action)
"""

if __name__ == '__main__':
    main()
