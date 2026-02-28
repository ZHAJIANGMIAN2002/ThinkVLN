

import habitat
import logging
import random
import json
import numpy as np
import argparse
import sys
import os
import torch
import multiprocessing as mp
from omegaconf import OmegaConf

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config import read_write
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    FogOfWarConfig,
)

from habitat.core.utils import try_cv2_import
from habitat_extensions import measures

cv2 = try_cv2_import()

DATASET = "r2r"
CONFIG_PATH = "./config/vln_r2r.yaml"
OUTPUT_PATH = f"./data/trajectory_data/{DATASET}"
DATA_PATH = None  # Set to None to use default dataset path

class StreamVLNHabitatRunner:
    def __init__(
        self,
        dataset: str,
        config_path: str,
        output_path: str,
        data_path: str = None,
        scenes_dir: str = None,
        save_frame_images: bool = False,
        frame_output_dir: str = None,
    ):
        self.device = torch.device("cuda")
        self.dataset = dataset.lower()
        self.config_path = config_path
        self.output_path = output_path
        self.data_path = data_path
        self.scenes_dir = scenes_dir
        self.save_frame_images = save_frame_images
        self.frame_output_dir = frame_output_dir

        self.config = get_habitat_config(self.config_path)
        
        # Add top-down map configuration
        with read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                }
            )

    def config_env(self, scene: str = None) -> habitat.Env:
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.update({"data_path": self.data_path})
        if self.scenes_dir is not None:
            with read_write(self.config):
                self.config.habitat.dataset.update({"scenes_dir": self.scenes_dir})
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)

    def generate(self, rank: int = 0, world_size: int = 1) -> None:
        os.makedirs(os.path.join(self.output_path), exist_ok=True)
        env = self.config_env()

        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        # Load already processed episodes from summary.json
        summary_file = os.path.join(self.output_path, "summary.json")
        processed_keys = set()
        
        if os.path.exists(summary_file):
            print(f"Loading previously processed episodes from {summary_file}")
            with open(summary_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Use scene_id + id as unique key
                        key = f"{data['scene_id']}_{data['id']}"
                        processed_keys.add(key)
                    except json.JSONDecodeError:
                        continue
            print(f"Found {len(processed_keys)} previously processed episodes")

        annotations = []
        for scene_id in sorted(scene_episode_dict.keys()):
            scan = scene_id.split("/")[-2]
            episodes = scene_episode_dict[scene_id]
            print(f"scene_id: {scene_id}, scan: {scan}")

            for episode in episodes[rank::world_size]:
                episode_id = int(episode.episode_id)
                scene_id = episode.scene_id.split('/')[-2]
                
                # Use scene_id + episode_id as unique key
                unique_key = f"{scene_id}_{episode_id}"
                
                # Skip if already processed
                if unique_key in processed_keys:
                    print(f"Skipping episode {scene_id}_{episode_id} (already processed)")
                    continue
                
                env.current_episode = episode
                agent = ShortestPathFollower(
                    sim=env.sim, goal_radius=0.5, return_one_hot=False)

                instructions = episode.instruction.instruction_text
                trajectory_id = episode.trajectory_id
                ref_path = episode.reference_path

                observation = env.reset()

                # episode initialization
                actions = [-1]
                next_waypoint_id = 1
                vis_frames = []
                if self.save_frame_images and self.frame_output_dir:
                    episode_frame_dir = os.path.join(
                        self.frame_output_dir, self.dataset,
                        f"{scene_id}_{self.dataset}_{episode_id:06d}"
                    )
                    os.makedirs(episode_frame_dir, exist_ok=True)

                while not env.episode_over:
                    rgb = observation["rgb"]

                    # Compose visualization frame with top-down map
                    info = env.get_metrics()
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb': observation['rgb']}, info)
                        vis_frames.append(frame)
                        if self.save_frame_images and self.frame_output_dir:
                            frame_idx = len(vis_frames) - 1
                            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(
                                os.path.join(episode_frame_dir, f"{frame_idx:06d}_rgb.jpg"),
                                rgb_bgr,
                            )
                            rgb_w = rgb.shape[1]
                            map_frame = frame[:, rgb_w:, :] if frame.shape[1] > rgb_w else frame
                            map_bgr = cv2.cvtColor(map_frame, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(
                                os.path.join(episode_frame_dir, f"{frame_idx:06d}_map.jpg"),
                                map_bgr,
                            )

                    next_action = agent.get_next_action(
                        ref_path[next_waypoint_id])

                    force_episode_over = False
                    while next_action == 0:
                        next_waypoint_id += 1
                        if next_waypoint_id == len(ref_path) - 1:
                            agent = ShortestPathFollower(
                                sim=env.sim, goal_radius=0.25, return_one_hot=False)
                        if next_waypoint_id >= len(ref_path):
                            force_episode_over = True
                            break
                        next_action = agent.get_next_action(
                            ref_path[next_waypoint_id])

                    if force_episode_over:
                        break

                    observation = env.step(next_action)
                    actions.append(next_action)

                if len(actions) > 498:
                    continue  # Skip episodes with too many actions

                assert len(actions) == len(
                    vis_frames), f"Actions length {len(actions)} does not match frames length {len(vis_frames)}"
                
                # Generate video from collected frames
                if len(vis_frames) > 0:
                    video_dir = os.path.join(
                        self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}"
                    )
                    os.makedirs(video_dir, exist_ok=True)
                    images_to_video(
                        vis_frames,
                        video_dir,
                        "trajectory",
                        fps=6,
                        quality=9
                    )
            

                result = {
                    "id": episode_id,
                    "key": unique_key,
                    "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                    "instructions": instructions if isinstance(instructions, list) else [instructions],
                    "actions": actions,
                    "trajectory_id": trajectory_id,
                    "scene_id": scene_id,
                }
                
                # Append to summary.json
                with open(os.path.join(self.output_path, "summary.json"), "a") as f:
                    f.write(json.dumps(result) + "\n")
                
                print(f"Processed episode {episode_id}")




def worker(rank, world_size, args):
    """The function that each spawned process will execute."""
    print(f"Starting worker with rank: {rank}")
    runner = StreamVLNHabitatRunner(
        dataset=args.dataset,
        config_path=args.config_path,
        output_path=args.output_path,
        data_path=args.data_path,
        scenes_dir=args.scenes_dir,
        save_frame_images=args.save_frame_images,
        frame_output_dir=args.frame_output_dir,
    )
    # 每个进程调用相同的 generate 函数，但传入不同的 rank
    runner.generate(rank=rank, world_size=world_size)
    print(f"Worker {rank} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--scenes_dir", type=str, default=None)
    parser.add_argument("--save_frame_images", action="store_true")
    parser.add_argument("--frame_output_dir", type=str, default=None)
    parser.add_argument(
        "--world_size", type=int, default=1, help="Number of concurrent processes to use."
    )
    args = parser.parse_args()

    # # Original multi-GPU (SLURM) version
    # # world_size = os.environ['SLURM_NTASKS']
    # # node_id = os.environ['SLURM_NODEID']
    # # rank = os.environ['SLURM_PROCID']
    # # local_rank = int(os.environ['SLURM_LOCALID'])
    # # node_list = os.environ['SLURM_NODELIST']
    #
    # # print(
    # #     f"rank: {rank}, world_size: {world_size}, node_id: {node_id}, local_rank: {local_rank}")
    #
    # # rank = int(rank)
    # # world_size = int(world_size)
    # # node_id = int(node_id)
    # # local_rank = int(local_rank)
    #
    # # runner = StreamVLNHabitatRunner(
    # #     dataset=args.dataset,
    # #     config_path=args.config_path,
    # #     output_path=args.output_path,
    # #     data_path=args.data_path
    # # )
    # # runner.generate(rank, world_size)
    # # print(f"Trajectory generation completed. rank: {rank}, world_size: {world_size}")

    # # Single GPU version
    # runner = StreamVLNHabitatRunner(
    #     dataset=args.dataset,
    #     config_path=args.config_path,
    #     output_path=args.output_path,
    #     data_path=args.data_path
    # )
    # runner.generate(rank=0, world_size=1)



    # 如果 world_size 为 1, 则沿用原有的单进程模式
    if args.world_size <= 1:
        print("Running in single-process mode.")
        runner = StreamVLNHabitatRunner(
            dataset=args.dataset,
            config_path=args.config_path,
            output_path=args.output_path,
            data_path=args.data_path,
            scenes_dir=args.scenes_dir,
            save_frame_images=args.save_frame_images,
            frame_output_dir=args.frame_output_dir,
        )
        runner.generate(rank=0, world_size=1)
        print("Trajectory generation completed.")
    # 如果 world_size > 1, 则启动多进程并发执行
    else:
        print(f"Running in multi-process mode with {args.world_size} workers.")
        
        # 'spawn' is safer for CUDA applications
        mp.set_start_method("spawn", force=True)

        processes = []
        for rank in range(args.world_size):
            # 现在 worker 是在顶层定义的，子进程可以找到它
            p = mp.Process(target=worker, args=(rank, args.world_size, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        print("All workers have completed trajectory generation.")