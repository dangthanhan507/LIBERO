from pointcloud import get_rgbd_image, get_point_cloud, get_colored_point_cloud
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T
from visualizer import Visualizer
from typing import Tuple, Union
from numpy import ndarray
from pathlib import Path
import numpy as np
import h5py
import json

def get_benchmark_demo_files(datasets_path: Path, dataset_name: str) -> list:
    benchmark_path = Path(datasets_path)/dataset_name
    demo_files = list(benchmark_path.rglob('*.hdf5'))
    return demo_files

def get_demo_states_and_env_metadata(demo_file: str, demo_name: str) -> Tuple[ndarray, dict]:
    with h5py.File(demo_file, "r") as f:
        env_metadata = json.loads(f["data"].attrs["env_args"])
        states = f['data'][demo_name]['states'][()]

    return states, env_metadata

def get_bddl_path(demo_file: Path, bddl_files_path: Union[str, Path] = None) -> Path:
    bddl_files_path = Path(bddl_files_path) if bddl_files_path else Path(get_libero_path("bddl_files"))
    bddl_path = bddl_files_path / demo_file.parent.stem / (demo_file.stem[:-5] + '.bddl')
    return bddl_path

def get_env_kwargs(env_metadata: dict, bddl_path: Union[str, Path]) -> dict:
    env_kwargs = env_metadata['env_kwargs']
    env_kwargs['controller'] = env_kwargs.pop('controller_configs')['type']
    env_kwargs['camera_depths'] = True
    env_kwargs['bddl_file_name'] = str(bddl_path)
    return env_kwargs

def get_rerendered_observations_and_intrinsics(states, env_kwargs: dict) -> dict:
    env = OffScreenRenderEnv(**env_kwargs)
    env.reset()

    camera_height, camera_width = env_kwargs['camera_heights'], env_kwargs['camera_widths']
    camera_names = env_kwargs['camera_names']
    observations = {}

    for camera_name in camera_names:
        key_camera_name = str(camera_name).replace('robot0_', '')
        obs_names = [f'{key_camera_name}_rgb', f'{key_camera_name}_depth', 'ee_states', 
                     'gripper_states', 'joint_states', 'ee_pos', 'ee_ori']

        for name in obs_names:
            observations[name] = []

        observations[f'{key_camera_name}_intrinsic'] = get_camera_intrinsic_matrix(env.sim, camera_name, camera_height, camera_width)
        observations[f'{key_camera_name}_position'] = get_camera_position(env.sim, camera_name),

    for state in states:
        observation = env.set_init_state(state)

        for camera_name in camera_names:
            key_camera_name = str(camera_name).replace('robot0_', '')
            rgb_img, depth_img = observation[camera_name + "_image"], observation[camera_name + "_depth"]

            observations[f'{key_camera_name}_rgb'].append(rgb_img)
            observations[f'{key_camera_name}_depth'].append(depth_img)

        ee_state = np.hstack((observation["robot0_eef_pos"], T.quat2axisangle(observation["robot0_eef_quat"]),))
        observations['ee_states'].append(ee_state)
        observations['ee_pos'].append(ee_state[:3])
        observations['ee_ori'].append(ee_state[3:])
        observations['gripper_states'].append(observation["robot0_gripper_qpos"])
        observations['joint_states'].append(observation["robot0_joint_pos"])

    for key in observations:
        if isinstance(observations[key], list):
            observations[key] = np.stack(observations[key], axis=0)

    env.close()
    return observations

def get_camera_position(sim, camera_name: str) -> ndarray:
    camera_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.model.body_pos[camera_id]
    return camera_pos

def verticalFlip(img: ndarray) -> ndarray:
    return np.flip(img, axis=0)

def get_args() -> Tuple[str, Path]:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_path", type=str, default=get_libero_path('datasets'))
    parser.add_argument("--dataset_name", type=str, default="libero_spatial")
    args = parser.parse_args()
    dataset_name, datasets_path = args.dataset_name, Path(args.datasets_path)

    return datasets_path, dataset_name

if __name__ == '__main__':
    datasets_path, dataset_name = get_args()
    demo_files = get_benchmark_demo_files(datasets_path, dataset_name)
    demo_file = demo_files[0]

    states, env_metadata = get_demo_states_and_env_metadata(demo_file, 'demo_0')
    bddl_path = get_bddl_path(demo_file)
    env_kwargs = get_env_kwargs(env_metadata, bddl_path)

    observations = get_rerendered_observations_and_intrinsics(states, env_kwargs)

    camera_names = env_kwargs['camera_names']
    camera_name = camera_names[1]
    camera_intrinsic = observations[f'{camera_name}_intrinsic']

    middle_time_step = len(observations[f'{camera_name}_rgb']) // 2
    rgb_image = observations[f'{camera_name}_rgb'][middle_time_step]
    depth_image = observations[f'{camera_name}_depth'][middle_time_step]

    rgbd_image = get_rgbd_image(rgb_image, depth_image)
    camera_height, camera_width = env_kwargs['camera_heights'], env_kwargs['camera_widths']
    point_cloud = get_point_cloud(rgbd_image, camera_intrinsic, camera_height, camera_width)
    colored_points = get_colored_point_cloud(point_cloud, rgb_image)
