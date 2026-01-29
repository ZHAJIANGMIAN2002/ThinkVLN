"""ThinkVLN Dataset and Collator for mixed action and CoT training"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any, Optional
import os
import json
import random

from ..tools.dataset_utils import (
    load_image, crop_cot_answer, extract_action_chunk, parse_frame_key
)


ACTION_MAPPING = {
    0: "stop",
    1: "forward",
    2: "turn_left",
    3: "turn_right"
}


class ThinkVLNDataset(Dataset):
    """Mixed action and CoT dataset for ThinkVLN training."""
    
    def __init__(
        self,
        action_data_path: Optional[str] = None,
        cot_data_path: Optional[str] = None,
        image_root: str = "/mnt/nvme/swx/dataset/R2R",
        seed: int = 42,
    ):
        self.action_data_path = action_data_path
        self.cot_data_path = cot_data_path
        self.image_root = image_root
        self.seed = seed
        
        self.action_samples = []
        self.cot_samples = []
        self.samples = []
        
        self._load_action_data()
        self._load_cot_data()
        self._mix_samples()
        
        print(f"Dataset: {len(self.action_samples)} action, "
              f"{len(self.cot_samples)} CoT, {len(self.samples)} total")
    
    def _load_action_data(self):
        """Load action trajectories and create frame-level samples."""
        if not self.action_data_path or not os.path.exists(self.action_data_path):
            return
        
        with open(self.action_data_path, 'r') as f:
            for line in f:
                traj = json.loads(line.strip())
                for frame_idx in range(traj['num_frames']):
                    subtask_idx = traj['subtask_sequence'][frame_idx]
                    plan_step = traj['plan'][subtask_idx - 1] if subtask_idx > 0 else traj['plan'][0]
                    
                    self.action_samples.append({
                        'data_type': 'action',
                        'episode_key': traj['episode_key'],
                        'frame_idx': frame_idx,
                        'instruction': traj['instruction'],
                        'current_plan_step': plan_step,
                        'current_subtask_idx': subtask_idx,
                        'actions': traj['actions'],
                        'subtask_sequence': traj['subtask_sequence'],
                    })
    
    def _load_cot_data(self):
        """Load CoT reasoning samples."""
        if not self.cot_data_path or not os.path.exists(self.cot_data_path):
            return
        
        with open(self.cot_data_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                
                # Parse plan and get current step
                plan_lines = [l.strip() for l in entry['plan'].split('\n') if l.strip()]
                plan_steps = []
                for line in plan_lines:
                    parts = line.split('.', 1)
                    plan_steps.append(parts[1].strip() if len(parts) > 1 else line.strip())
                
                subtask_idx = entry['ground_truth_subtask']
                plan_step = plan_steps[subtask_idx - 1] if 0 < subtask_idx <= len(plan_steps) else plan_steps[0]
                
                self.cot_samples.append({
                    'data_type': 'cot',
                    'frame_key': entry['frame_key'],
                    'episode_key': entry['episode_key'],
                    'instruction': entry['instruction'],
                    'current_plan_step': plan_step,
                    'answer': crop_cot_answer(entry['answer']),
                })
    
    def _mix_samples(self):
        """Mix action and CoT samples."""
        if not self.action_samples and not self.cot_samples:
            return
        
        random.seed(self.seed)
        self.samples = self.action_samples + self.cot_samples
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class ThinkVLNDataCollator:
    """Collator for processing mixed action and CoT samples."""
    
    def __init__(
        self,
        processor,
        num_query_tokens: int = 4,
        query_token_id: int = 151700,
        image_root: str = "/mnt/nvme/swx/dataset/R2R",
    ):
        self.processor = processor
        self.num_query_tokens = num_query_tokens
        self.query_token_id = query_token_id
        self.image_root = image_root
        
        self.action_prompt = "Based on the current observation and subgoal '{subgoal}', predict the next 4 actions."
        self.cot_prompt = "Based on the current observation and subgoal '{subgoal}', think step by step to determine the action."
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch of mixed samples."""
        processed = []
        
        for sample in batch:
            if sample['data_type'] == 'action':
                processed.append(self._process_action(sample))
            else:
                processed.append(self._process_cot(sample))
        
        return self._collate(processed)
    
    def _process_action(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process action sample with query tokens."""
        # Convert episode_key format: "17DRP5sb8fy_10154" -> "17DRP5sb8fy_r2r_010154"
        episode_key = sample['episode_key']
        parts = episode_key.split('_')
        if len(parts) == 2:
            scene_id = parts[0]
            episode_id = parts[1]
            dir_episode_key = f"{scene_id}_r2r_{int(episode_id):06d}"
        else:
            dir_episode_key = episode_key
        
        # Load image
        image_path = os.path.join(
            self.image_root, 
            dir_episode_key, 
            f"{sample['frame_idx']:06d}_rgb.jpg"
        )
        image = load_image(image_path)
        
        # Create prompt and process
        prompt = self.action_prompt.format(subgoal=sample['current_plan_step'])
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=False)
        
        # Extract and append query tokens
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        query_tokens = torch.full((self.num_query_tokens,), self.query_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, query_tokens])
        attention_mask = torch.cat([attention_mask, torch.ones(self.num_query_tokens, dtype=torch.long)])
        
        # Extract labels
        actions, progress = extract_action_chunk(
            sample['frame_idx'], 
            sample['actions'], 
            sample['subtask_sequence'], 
            self.num_query_tokens
        )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': inputs['pixel_values'][0] if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'][0] if 'image_grid_thw' in inputs else None,
            'action_labels': torch.tensor(actions, dtype=torch.long),
            'progress_labels': torch.tensor(progress, dtype=torch.float32),
            'labels': None,
            'data_type': 'action'
        }
    
    def _process_cot(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process CoT sample with label masking."""
        # Parse frame_key to get episode_key and step_id
        episode_key, step_id = parse_frame_key(sample['frame_key'])
        
        # Convert episode_key format: "17DRP5sb8fy_10154" -> "17DRP5sb8fy_r2r_010154"
        parts = episode_key.split('_')
        if len(parts) == 2:
            scene_id = parts[0]
            episode_id = parts[1]
            dir_episode_key = f"{scene_id}_r2r_{int(episode_id):06d}"
        else:
            dir_episode_key = episode_key
        
        # Load image
        image_path = os.path.join(self.image_root, dir_episode_key, f"{step_id:06d}_rgb.jpg")
        image = load_image(image_path)
        
        # Create prompt
        prompt = self.cot_prompt.format(subgoal=sample['current_plan_step'])
        full_text = prompt + "\n" + sample['answer']
        
        # Process prompt only to find mask boundary
        messages_prompt = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        text_prompt = self.processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
        inputs_prompt = self.processor(text=[text_prompt], images=[image], return_tensors="pt", padding=False)
        prompt_len = inputs_prompt['input_ids'].shape[1]
        
        # Process full text
        messages_full = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_text}
            ]
        }]
        text_full = self.processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_full], images=[image], return_tensors="pt", padding=False)
        
        # Create labels with prompt masked
        input_ids = inputs['input_ids'][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'][0],
            'pixel_values': inputs['pixel_values'][0] if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'][0] if 'image_grid_thw' in inputs else None,
            'action_labels': None,
            'progress_labels': None,
            'labels': labels,
            'data_type': 'cot'
        }
    
    def _collate(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate samples into padded batch."""
        batch_size = len(samples)
        max_len = max(s['input_ids'].shape[0] for s in samples)
        pad_id = self.processor.tokenizer.pad_token_id
        
        # Initialize tensors
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        has_action = any(s['action_labels'] is not None for s in samples)
        action_labels = torch.full((batch_size, self.num_query_tokens), -100, dtype=torch.long) if has_action else None
        progress_labels = torch.full((batch_size, self.num_query_tokens), -100.0, dtype=torch.float32) if has_action else None
        
        pixel_values_list = []
        image_grid_list = []
        
        # Fill batch
        for i, s in enumerate(samples):
            seq_len = s['input_ids'].shape[0]
            input_ids[i, :seq_len] = s['input_ids']
            attention_mask[i, :seq_len] = s['attention_mask']
            
            if s['data_type'] == 'cot':
                labels[i, :seq_len] = s['labels']
            elif has_action and s['action_labels'] is not None:
                action_labels[i] = s['action_labels']
                progress_labels[i] = s['progress_labels']
            
            if s['pixel_values'] is not None:
                pixel_values_list.append(s['pixel_values'])
            if s['image_grid_thw'] is not None:
                image_grid_list.append(s['image_grid_thw'])
        
        # Set None for uniform batches
        if all(s['data_type'] == 'action' for s in samples):
            labels = None
        if all(s['data_type'] == 'cot' for s in samples):
            action_labels = None
            progress_labels = None
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': torch.stack(pixel_values_list) if pixel_values_list else None,
            'image_grid_thw': torch.stack(image_grid_list) if image_grid_list else None,
            'action_labels': action_labels,
            'progress_labels': progress_labels,
            'labels': labels,
        }
