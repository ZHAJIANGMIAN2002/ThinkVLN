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

from thinkvln.engine.inference import load_model_and_processor, run_batch_inference
import torch.distributed as dist


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        processor: Any = None,
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

        self.model = model
        self.processor = processor
        
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

    def build_user_message(
        self,
        instruction: str,
        plan: str,
        prev_subtask: Optional[str] = None
    ) -> str:
        """
        Build user message in clean format.
        
        Args:
            instruction: Navigation instruction text
            plan: Step-by-step plan as string
            prev_subtask: Previous subtask information (optional)
        
        Returns:
            Formatted user message string
        """
        user_message = f"<image>\n**Instruction**: {instruction}\n\n**Plan**: {plan}"
        
        if prev_subtask:
            user_message += f"\n**Previous Subtask**: {prev_subtask}"
        
        return user_message

    def parse_action(self, output: str) -> int:
        """
        Parse action from model CoT output.
        
        Args:
            output: Model output text containing [action] section
        
        Returns:
            Action ID (0=stop, 1=forward, 2=turn_left, 3=turn_right)
        """
        # Look for [action] section
        if "[action]" in output.lower():
            action_start = output.lower().find("[action]")
            action_lines = output[action_start:].split('\n')
            
            # Search for action value in lines after [action]
            for line in action_lines[1:]:
                line_lower = line.strip().lower()
                
                # Check for exact matches first
                if line_lower in ['forward', 'turn_left', 'turn_right', 'stop']:
                    return self.actions2idx[line_lower]
                # Check for partial matches
                elif 'forward' in line_lower:
                    return self.actions2idx['forward']
                elif 'turn_left' in line_lower or 'turn left' in line_lower:
                    return self.actions2idx['turn_left']
                elif 'turn_right' in line_lower or 'turn right' in line_lower:
                    return self.actions2idx['turn_right']
                elif 'stop' in line_lower:
                    return self.actions2idx['stop']
        
        # Default to stop if action not found
        return self.actions2idx['stop']

    def extract_prev_subtask_from_output(self, output: str) -> Optional[str]:
        """
        Extract prev subtask from model output for next step's use.
        
        Args:
            output: Model output text
        
        Returns:
            Previous subtask as string or None
        """
        # Look for [subtask determination] or [cur subtask] section
        if "[cur subtask]" in output.lower():
            subtask_start = output.lower().find("[cur subtask]")
            subtask_text = output[subtask_start:].split('\n')[0]
            # Extract content after "[cur subtask]"
            content = subtask_text.split("]", 1)[-1].strip()
            if content:
                return content
        
        if "[subtask determination]" in output.lower():
            subtask_start = output.lower().find("[cur subtask]")
            if subtask_start != -1:
                subtask_text = output[subtask_start:].split('\n')[0]
                content = subtask_text.split("]", 1)[-1].strip()
                if content:
                    return content
        
        return None

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
                        done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                        if get_rank() == 0:
                            sucs.append(res['success'])
                            spls.append(res['spl'])
                            oss.append(res['os'])
                            ones.append(res['ne'])
                    except json.JSONDecodeError:
                        continue

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
                
                while not env.episode_over:
                    self.model.eval()
                    
                    # Get current observation
                    rgb = observations["rgb"]
                    info = env.get_metrics()
                    
                    # Prepare concatenated image (RGB + top-down map)
                    image = self.prepare_image_with_map(rgb, info)
                    
                    # Build prompt
                    # Extract plan from episode if available
                    plan = "1. Navigate to the goal."
                    if hasattr(episode, 'reference_path'):
                        # Use a simple plan if available
                        plan = "1. Follow the reference path."
                    
                    user_message = self.build_user_message(
                        instruction=episode_instruction,
                        plan=plan,
                        prev_subtask=prev_subtask
                    )
                    
                    # Run inference
                    print(f"Step {step_id}: Generating action...")
                    try:
                        llm_outputs = run_batch_inference(
                            self.model,
                            self.processor,
                            [user_message],
                            [image],
                            self.device,
                            max_new_tokens=1024
                        )[0]
                        
                        print(f"Model output:\n{llm_outputs}", flush=True)
                    except Exception as e:
                        print(f"Error during inference: {e}")
                        llm_outputs = "[action]\nstop"
                    
                    # Parse action
                    action = self.parse_action(llm_outputs)
                    print(f"Parsed action: {self.idx2actions[action]}", flush=True)
                    
                    # Extract previous subtask for next step
                    prev_subtask = self.extract_prev_subtask_from_output(llm_outputs)
                    
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
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/env_eval')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help="Maximum sequence length for model input")
    
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

    # Load tokenizer and model
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model_and_processor(args.model_path, device)
    model.requires_grad_(False)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Run evaluation (pass rank and world_size to avoid re-initializing)
    evaluate(model, processor, args, rank, world_size, gpu)


def evaluate(model, processor, args, rank, world_size, gpu):
    """Run evaluation on all episodes."""
    model.eval()
    
    # Don't re-initialize distributed mode - it's already done in eval()
    
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        processor=processor,
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

