from typing import Callable
import os

import pickle

from rozumarm_vima_utils.cv.camera import Camera, CamDenseReader
from rozumarm_vima_utils.rozumarm_vima_cv.segment_scene import segment_scene

import numpy as np
import cv2
from time import sleep

from rozumarm_vima_utils.transform import rf_tf_c2r, map_tf_repr_c2r
from rozumarm_vima_utils.vima.vima_model import VimaModel
# from rozumarm_vima_utils.rudolph_model import RuDolphModel
import argparse

def run_loop(r, robot, oracle, cubes_detector, model=None, n_iters=3):
    """
    r: scene renderer
    """
    while True:
        for i in range(n_iters):
            n_cubes = -1
            while n_cubes != 2:
                obj_posquats = cubes_detector.detect()
                n_cubes = len(obj_posquats)

            # map from cam to rozum
            obj_posquats = [
                (rf_tf_c2r(pos), map_tf_repr_c2r(quat)) for pos, quat in obj_posquats
            ]

            front_img, top_img = r.render_scene(obj_posquats)
            obs = {
                'rgb': {
                    'front': np.transpose(front_img, axes=(2, 0, 1)),
                    'top': np.transpose(top_img, axes=(2, 0, 1))
                },
                'segm': r.env._get_obs()['segm'],
                'ee': 1  # spatula
            }

            meta_info = {'action_bounds':{'low': np.array([ 0.25, -0.5 ]), 'high': np.array([0.75, 0.5 ])}}
            
            meta_info["n_objects"] = 4
            meta_info["obj_id_to_info"] = {4: {'obj_name': 'three-sided rectangle'},
                                            5: {'obj_name': 'line'},
                                            6: {'obj_name': 'small block'},
                                            7: {'obj_name': 'small block'}}
            model.reset(r.env.prompt,r.env.prompt_assets)
            #action = model.step(obs,meta_info)
            action = oracle.act(obs)
            if action is None:
                print("Press Enter to try again, or q + Enter to exit.")
                ret = input()
                if len(ret) > 0 and ret[0] == 'q':
                    return
                r.reset(exact_num_swept_objects=1)
                continue
                # print("ORACLE FAILED.")
                # # cubes_detector.release()
                # return

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

        print("Press Enter to try again, or q + Enter to exit.")
        ret = input()
        if len(ret) > 0 and ret[0] == 'q':
            return
        r.reset(exact_num_swept_objects=1)
        model.reset(r.env.prompt,r.env.prompt_assets)
        continue


def run_loop_sim_to_real(r, prompt_assets, robot, oracle, model=None, n_iters=3):
    """
    r: scene renderer
    """

    cam_1 = CamDenseReader(3, 'cam_top_video.mp4')
    cam_2 = CamDenseReader(2, 'cam_front_video.mp4')
    cam_1.start_recording()
    cam_2.start_recording()
    sleep(3)

    counter = 1
    while True:
        counter += 1
        for i in range(n_iters):
            _, image_top = cam_1.read_image()
            _, image_front = cam_2.read_image()

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

            with open(f"model_input_{counter}.pickle", 'wb') as f:
                pickle.dump({'obs': obs, 'prompt_assets': prompt_assets}, f, protocol=pickle.HIGHEST_PROTOCOL)

            meta_info = {'action_bounds':{'low': np.array([ 0.25, -0.5 ]), 'high': np.array([0.75, 0.5 ])}}
            
            meta_info["n_objects"] = 4
            meta_info["obj_id_to_info"] = {4: {'obj_name': 'three-sided rectangle'},
                                            5: {'obj_name': 'line'},
                                            6: {'obj_name': 'small block'},
                                            7: {'obj_name': 'small block'}}
            model.reset(r.env.prompt, prompt_assets)
            action = model.step(obs,meta_info)
            #action = oracle.act(obs)
            if action is None:
                print("Press Enter to try again, or q + Enter to exit.")
                ret = input()
                if len(ret) > 0 and ret[0] == 'q':
                    cam_1.stop_recording()
                    cam_2.stop_recording()
                    return
                r.reset(exact_num_swept_objects=1)
                continue
                # print("ORACLE FAILED.")
                # # cubes_detector.release()
                # return

            clipped_action = {
                k: np.clip(v, r.env.action_space[k].low, r.env.action_space[k].high)
                for k, v in action.items()
            }

            pos_0 = clipped_action["pose0_position"]
            pos_1 = clipped_action["pose1_position"]
            eef_quat = robot.get_swipe_quat(pos_0, pos_1)
            
            x_compensation_bias = 0.03
            pos_0[0] += x_compensation_bias
            pos_1[0] += x_compensation_bias
            
            posquat_0 = (pos_0, eef_quat)
            posquat_1 = (pos_1, eef_quat)
            robot.swipe(posquat_0, posquat_1)

        print("Press Enter to start over...")
        ret = input()
        if len(ret) > 0 and ret[0] == 'q':
            cam_1.stop_recording()
            cam_2.stop_recording()
            return
        r.reset(exact_num_swept_objects=1)
        model.reset(r.env.prompt,r.env.prompt_assets)
        continue


def get_prompt_assets():
    folder = "/home/daniil/code/rozumarm-vima-utils/rozumarm_vima_utils/rozumarm_vima_cv/images/prompts/"

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


from rozumarm_vima_utils.scene_renderer import VIMASceneRenderer
def main():
    from rozumarm_vima_utils.robot import RozumArm
    from rozumarm_vima_utils.scripts.detect_cubes import mock_detect_cubes
    # from rozumarm_vima_utils.cv.test import CubeDetector

    r = VIMASceneRenderer('sweep_without_exceeding')
    oracle = r.env.task.oracle(r.env)
    robot = RozumArm(use_mock_api=False)

    arg = argparse.ArgumentParser()
    arg.add_argument("--partition", type=str, default="placement_generalization")
    arg.add_argument("--task", type=str, default="visual_manipulation")
    arg.add_argument("--ckpt", type=str, required=True)
    arg.add_argument("--device", default="cpu")
    arg = arg.parse_args("--ckpt /home/daniil/code/rozumarm-vima-utils/2M.ckpt --device cuda --task sweep_without_exceeding".split())    
    #assert False, os.listdir()
    model = VimaModel(arg)
    #model = RuDolphModel()

    r.reset(exact_num_swept_objects=1)
    # prompt_assets = get_prompt_assets()
    # model.reset(r.env.prompt, prompt_assets)
    model.reset(r.env.prompt, r.env.prompt_assets)
    
    from rozumarm_vima_utils.cv.test import detector
    # detector = CubeDetector()
    run_loop(r, robot, oracle, cubes_detector=detector, model=model, n_iters=1)
    # run_loop_sim_to_real(r, prompt_assets, robot, oracle, model=model, n_iters=2)
    detector.release()


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
