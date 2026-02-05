#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ThinkVLN Environment Evaluation Script

This script evaluates the ThinkVLN model in the Habitat simulator environment.
At each step, it processes the environment state (RGB + top-down map), generates
navigation actions using the model's CoT reasoning, and computes evaluation metrics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import re
import json
import torch
import random
import argparse
import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm

import habitat
from habitat import logger, Env
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from thinkvln.habitat_extensions import measures

from thinkvln.engine.inference import load_model_and_processor
from thinkvln.models.navigation_model import NavigationModel, ThinkVLNNavigationModel, StreamVLNNavigationModel
import torch.distributed as dist


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        nav_model: NavigationModel = None,  # Accept NavigationModel wrapper
        model: Any = None,  # Keep for backward compatibility
        processor: Any = None,  # Keep for backward compatibility
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        
        # Store model type for conditional processing
        self.model_type = getattr(args, 'model_type', 'thinkvln') if args else 'thinkvln'

        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
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
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        # Use provided NavigationModel wrapper, or create one from model/processor
        if nav_model is not None:
            self.nav_model = nav_model
        elif model is not None and processor is not None:
            # Backward compatibility: create ThinkVLN wrapper
            self.nav_model = ThinkVLNNavigationModel(
                model=model,
                processor=processor,
                device=str(self.device),
                max_new_tokens=getattr(args, 'model_max_length', 4096) if args else 1024
            )
        else:
            self.nav_model = None
        
        # Keep for backward compatibility and action name mapping
        self.actions2idx = OrderedDict({
            'stop': 0,
            'forward': 1,
            'turn_left': 2,
            'turn_right': 3
        })
        self.idx2actions = {v: k for k, v in self.actions2idx.items()}

    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def prepare_image_with_map(self, rgb: np.ndarray, info: Dict) -> Image.Image:
        """
        Concatenate RGB image (left) with top-down map (right).
        
        Args:
            rgb: RGB observation numpy array (H, W, 3)
            info: Environment info dict containing 'top_down_map'
        
        Returns:
            PIL Image of concatenated RGB and top-down map
        """
        rgb_image = rgb.astype(np.uint8)
        
        # Get top-down map and colorize it to match RGB height
        if info.get('top_down_map') is not None:
            top_down_map = info['top_down_map']
            top_down_map_vis = maps.colorize_draw_agent_and_fit_to_height(
                top_down_map, rgb_image.shape[0]
            )
        else:
            # If no top-down map, create a placeholder
            top_down_map_vis = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)
        
        # Concatenate horizontally (RGB left, map right)
        concatenated = np.concatenate((rgb_image, top_down_map_vis), axis=1)
        
        # Convert to PIL Image
        image = Image.fromarray(concatenated)
        return image


    def eval_action(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run evaluation on episodes assigned to this worker.
        
        Args:
            idx: Worker index for distributed evaluation
        
        Returns:
            Tuple of (success, spl, oracle_success, distance_to_goal, num_episodes)
        """
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        
        # Load already processed results if they exist
        result_file = os.path.join(self.output_path, f'result.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                for line in f.readlines():
                    try:
                        res = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Skip summary or malformed lines that don't have per-episode fields
                    if not all(k in res for k in ["scene_id", "episode_id", "episode_instruction"]):
                        continue
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res.get('success', 0.0))
                        spls.append(res.get('spl', 0.0))
                        oss.append(res.get('os', 0.0))
                        ones.append(res.get('ne', 0.0))

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            
            process_bar = tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start", episode_instruction)
                episode_id = episode.episode_id
                
                # Skip if already processed
                if [scene_id, episode_id, episode_instruction] in done_res:
                    process_bar.update(1)
                    continue
                
                env.current_episode = episode
                observations = env.reset()
                
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(
                    os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg')
                )
                
                step_id = 0
                prev_subtask = None
                
                # Initialize episode-specific state for StreamVLN
                if self.model_type == 'streamvln' and isinstance(self.nav_model, StreamVLNNavigationModel):
                    # Reset StreamVLN state for new episode
                    self.nav_model.rgb_list = []
                    self.nav_model.depth_list = []
                    self.nav_model.pose_list = []
                    self.nav_model.intrinsic_list = []
                    self.nav_model.time_ids = []
                    self.nav_model.action_seq = []
                    self.nav_model.past_key_values = None
                    self.nav_model.output_ids = None
                    self.nav_model.step_count = 0
                    initial_height = env.sim.get_agent_state().position[1]
                
                while not env.episode_over:
                    if self.nav_model is None:
                        raise ValueError("Navigation model not initialized")
                    
                    self.nav_model.eval()
                    
                    # Get current observation
                    rgb = observations["rgb"]
                    info = env.get_metrics()
                    
                    # Prepare input based on model type
                    if self.model_type == 'streamvln' and isinstance(self.nav_model, StreamVLNNavigationModel):
                        # StreamVLN needs full observations dict with multi-modal data
                        observation_dict = {
                            'rgb': rgb,
                            'depth': observations.get('depth'),
                            'gps': observations.get('gps', [0, 0]),
                            'compass': observations.get('compass', [0]),
                            'env': env,
                            'sensor_config': self.sim_sensors_config,
                            'initial_height': initial_height,
                            'camera_height': self.sim_sensors_config.rgb_sensor.position[1],
                            'min_depth': self.sim_sensors_config.depth_sensor.min_depth,
                            'max_depth': self.sim_sensors_config.depth_sensor.max_depth,
                        }
                        observation_input = observation_dict
                    else:
                        # ThinkVLN uses concatenated image (RGB + top-down map)
                        image = self.prepare_image_with_map(rgb, info)
                        observation_input = image
                    
                    # Extract plan from episode if available
                    plan = "1. Navigate to the goal."
                    if hasattr(episode, 'reference_path'):
                        plan = "1. Follow the reference path."
                    
                    # Call model to predict action (encapsulates all model-specific logic)
                    action, prev_subtask = self.nav_model.predict_action(
                        observation=observation_input,
                        instruction=episode_instruction,
                        plan=plan,
                        prev_subtask=prev_subtask
                    )
                    
                    # Execute action
                    observations = env.step(action)
                    step_id += 1
                
                # Episode finished
                metrics = env.get_metrics()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, "
                      f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                      f"ne: {metrics['distance_to_goal']}")
                
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                
                with open(result_file, 'a') as f:
                    f.write(json.dumps(result) + "\n")
                
                process_bar.update(1)

        env.close()
        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(ones).to(self.device),
            torch.tensor(len(sucs)).to(self.device)
        )


def init_dist_mode():
    """Initialize distributed mode for multi-GPU evaluation."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
    else:
        rank = 0
        world_size = 1
        gpu = 0
    return rank, world_size, gpu


def get_rank():
    """Get current process rank."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get total number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", default=0, type=int, dest="local_rank", help="node rank")
    parser.add_argument("--model_path", type=str, default="", help="Path to model")
    parser.add_argument("--model_type", type=str, default="thinkvln", choices=["thinkvln", "streamvln"],
                        help="Model type: thinkvln or streamvln")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/env_eval')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help="Maximum sequence length for model input")
    
    # StreamVLN specific parameters
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Number of frames before resetting (StreamVLN)")
    parser.add_argument("--num_future_steps", type=int, default=4,
                        help="Future steps for history sampling (StreamVLN)")
    parser.add_argument("--num_history", type=int, default=8,
                        help="Number of history frames to use (StreamVLN)")
    
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    
    args = parser.parse_args()
    
    # Initialize distributed mode ONCE at the beginning
    rank, world_size, gpu = init_dist_mode()
    local_rank = args.local_rank

    # Set device
    device = f"cuda:{gpu}" if world_size > 1 else args.device

    # Load model based on type
    print(f"Loading {args.model_type} model from {args.model_path}...")
    
    if args.model_type == "thinkvln":
        model, processor = load_model_and_processor(args.model_path, device)
        model.requires_grad_(False)
        model.eval()
        nav_model = ThinkVLNNavigationModel(
            model=model,
            processor=processor,
            device=str(device),
            max_new_tokens=args.model_max_length
        )
    elif args.model_type == "streamvln":
        # StreamVLN requires LLaVA-NeXT codebase
        # Try to find and add LLaVA-NeXT to Python path
        import sys
        thinkvln_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Possible LLaVA-NeXT paths
        possible_llava_paths = [
            os.path.join(thinkvln_root, "third_party", "LLaVA-NeXT"),
            os.path.join(thinkvln_root, "LLaVA-NeXT"),
            os.path.expanduser("~/LLaVA-NeXT"),
        ]
        
        # Add ThinkVLN root first (for streamvln imports)
        if thinkvln_root not in sys.path:
            sys.path.insert(0, thinkvln_root)
        
        # Try to find and add LLaVA-NeXT
        llava_found = False
        for llava_path in possible_llava_paths:
            if os.path.exists(llava_path) and os.path.exists(os.path.join(llava_path, "llava")):
                if llava_path not in sys.path:
                    sys.path.insert(0, llava_path)
                
                # Apply compatibility patch for transformers version issues
                compat_patch_path = os.path.join(llava_path, "llava", "compat_patch.py")
                if os.path.exists(compat_patch_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("llava.compat_patch", compat_patch_path)
                    compat_patch = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(compat_patch)
                    if rank == 0:
                        print(f"✓ Applied compatibility patch for transformers")
                
                llava_found = True
                if rank == 0:
                    print(f"✓ Found LLaVA-NeXT at: {llava_path}")
                break
        
        if not llava_found:
            raise ImportError("LLaVA-NeXT not found. StreamVLN requires LLaVA-NeXT codebase.")
        
        import transformers
        from streamvln.model.stream_video_vln import StreamVLNForCausalLM
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=args.model_max_length,
            padding_side="right"
        )
        config = transformers.AutoConfig.from_pretrained(args.model_path)
        
        # Fix compatibility: Add layer_types if missing (required by newer transformers)
        if not hasattr(config, 'layer_types') or config.layer_types is None:
            # Generate layer_types based on Qwen2Config logic
            num_layers = getattr(config, 'num_hidden_layers', 32)
            sliding_window = getattr(config, 'sliding_window', None)
            max_window_layers = getattr(config, 'max_window_layers', num_layers)
            
            if sliding_window is not None:
                config.layer_types = [
                    "sliding_attention" if i >= max_window_layers else "full_attention"
                    for i in range(num_layers)
                ]
            else:
                config.layer_types = ["full_attention"] * num_layers
        
        model = StreamVLNForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=False,
        )
        model.model.num_history = args.num_history
        model.requires_grad_(False)
        model.to(device)
        model.eval()
        
        # Initialize StreamVLN model state for distributed evaluation
        # This must be called before creating NavigationModel wrapper
        model.reset(world_size)
        
        nav_model = StreamVLNNavigationModel(
            model=model,
            tokenizer=tokenizer,
            device=str(device),
            num_frames=args.num_frames,
            num_future_steps=args.num_future_steps,
            num_history=args.num_history,
            env_id=rank
        )
        processor = None  # StreamVLN doesn't use processor
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Run evaluation
    evaluate(nav_model, args, rank, world_size, gpu)


def evaluate(nav_model, args, rank, world_size, gpu):
    """Run evaluation on all episodes."""
    nav_model.eval()
    
    # Don't re-initialize distributed mode - it's already done in eval()
    
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        nav_model=nav_model,  # Pass NavigationModel wrapper directly
        epoch=0,
        args=args
    )
    
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(rank)
    
    # Gather results from all ranks
    if world_size > 1:
        ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
        dist.all_gather(ep_num_all, ep_num)
        
        sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
        spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
        oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
        ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
        
        dist.barrier()
        dist.all_gather(sucs_all, sucs)
        dist.all_gather(spls_all, spls)
        dist.all_gather(oss_all, oss)
        dist.all_gather(ones_all, ones)
        dist.barrier()
        
        sucs_all = torch.cat(sucs_all, dim=0)
        spls_all = torch.cat(spls_all, dim=0)
        oss_all = torch.cat(oss_all, dim=0)
        ones_all = torch.cat(ones_all, dim=0)
    else:
        sucs_all = sucs
        spls_all = spls
        oss_all = oss
        ones_all = ones
    
    # Compute summary statistics
    if rank == 0:
        result_all = {
            "sucs_all": (sum(sucs_all) / len(sucs_all)).item() if len(sucs_all) > 0 else 0.0,
            "spls_all": (sum(spls_all) / len(spls_all)).item() if len(spls_all) > 0 else 0.0,
            "oss_all": (sum(oss_all) / len(oss_all)).item() if len(oss_all) > 0 else 0.0,
            "ones_all": (sum(ones_all) / len(ones_all)).item() if len(ones_all) > 0 else 0.0,
            'length': len(sucs_all)
        }
        
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Total episodes: {result_all['length']}")
        print(f"Success Rate: {result_all['sucs_all']:.2%}")
        print(f"SPL: {result_all['spls_all']:.4f}")
        print(f"Oracle Success: {result_all['oss_all']:.2%}")
        print(f"Distance to Goal: {result_all['ones_all']:.4f}")
        print("=" * 80)
        
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all) + "\n")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    eval()

