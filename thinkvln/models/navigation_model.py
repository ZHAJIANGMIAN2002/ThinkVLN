#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Navigation Model Interface

This module defines a unified interface for navigation models.
All navigation models should implement this interface to work with the evaluator.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
from PIL import Image
import torch
import numpy as np


class NavigationModel(ABC):
    """
    Abstract base class for navigation models.
    
    This interface ensures that evaluators can work with any model implementation
    without needing to know model-specific details.
    """
    
    @abstractmethod
    def predict_action(
        self,
        observation: Image.Image,
        instruction: str,
        plan: Optional[str] = None,
        prev_subtask: Optional[str] = None,
        **kwargs
    ) -> Tuple[int, Optional[str]]:
        """
        Predict action from observation and instruction.
        
        Args:
            observation: Current observation (RGB + map image)
            instruction: Navigation instruction text
            plan: Step-by-step plan (optional)
            prev_subtask: Previous subtask information (optional)
            **kwargs: Additional model-specific arguments
        
        Returns:
            Tuple of (action_index, prev_subtask):
            - action_index: Action ID (0=stop, 1=forward, 2=turn_left, 3=turn_right)
            - prev_subtask: Current subtask string for next step (optional)
        """
        pass
    
    @abstractmethod
    def eval(self):
        """Set model to evaluation mode."""
        pass


class ThinkVLNNavigationModel(NavigationModel):
    """
    ThinkVLN model wrapper implementing NavigationModel interface.
    
    This wrapper encapsulates all ThinkVLN-specific logic (inference, parsing, etc.)
    so that the evaluator doesn't need to know about these details.
    """
    
    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        max_new_tokens: int = 1024
    ):
        """
        Initialize ThinkVLN navigation model wrapper.
        
        Args:
            model: The ThinkVLN model instance
            processor: The processor for the model
            device: Device to use for inference
            max_new_tokens: Maximum tokens to generate
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Action mapping
        self.actions2idx = {
            'stop': 0,
            'forward': 1,
            'turn_left': 2,
            'turn_right': 3
        }
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def _build_user_message(
        self,
        instruction: str,
        plan: str,
        prev_subtask: Optional[str] = None
    ) -> str:
        """Build user message in clean format."""
        user_message = f"<image>\n**Instruction**: {instruction}\n\n**Plan**: {plan}"
        
        if prev_subtask:
            user_message += f"\n**Previous Subtask**: {prev_subtask}"
        
        return user_message
    
    def _parse_action(self, output: str) -> int:
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
    
    def _extract_prev_subtask(self, output: str) -> Optional[str]:
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
    
    def predict_action(
        self,
        observation: Image.Image,
        instruction: str,
        plan: Optional[str] = None,
        prev_subtask: Optional[str] = None,
        **kwargs
    ) -> Tuple[int, Optional[str]]:
        """
        Predict action from observation and instruction.
        
        This method encapsulates all ThinkVLN-specific logic:
        - Building the prompt
        - Running inference
        - Parsing the action
        - Extracting subtask information
        """
        from thinkvln.engine.inference import run_batch_inference
        
        # Build plan if not provided
        if plan is None:
            plan = "1. Navigate to the goal."
        
        # Build user message
        user_message = self._build_user_message(
            instruction=instruction,
            plan=plan,
            prev_subtask=prev_subtask
        )
        
        # Run inference
        try:
            llm_outputs = run_batch_inference(
                self.model,
                self.processor,
                [user_message],
                [observation],
                self.device,
                max_new_tokens=self.max_new_tokens
            )[0]
        except Exception as e:
            print(f"Error during inference: {e}")
            llm_outputs = "[action]\nstop"
        
        # Parse action
        action = self._parse_action(llm_outputs)
        
        # Extract previous subtask for next step
        prev_subtask = self._extract_prev_subtask(llm_outputs)
        
        return action, prev_subtask


class StreamVLNNavigationModel(NavigationModel):
    """
    StreamVLN model wrapper implementing NavigationModel interface.
    
    This wrapper encapsulates all StreamVLN-specific logic (multi-modal processing,
    video sequence management, action sequence generation, etc.)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        num_frames: int = 32,
        num_future_steps: int = 4,
        num_history: int = 8,
        env_id: int = 0
    ):
        """
        Initialize StreamVLN navigation model wrapper.
        
        Args:
            model: The StreamVLN model instance
            tokenizer: The tokenizer for the model
            device: Device to use for inference
            num_frames: Number of frames before resetting
            num_future_steps: Future steps for history sampling
            num_history: Number of history frames to use
            env_id: Environment ID for model reset
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_frames = num_frames
        self.num_future_steps = num_future_steps
        self.num_history = num_history
        self.env_id = env_id
        
        # StreamVLN action mapping (different from ThinkVLN)
        self.actions2idx = {
            'STOP': 0,
            "↑": 1,    # forward
            "←": 2,    # turn_left
            "→": 3     # turn_right
        }
        
        # Initialize conversation template
        self.conversation = [
            {"from": "human", "value": "<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."},
            {"from": "gpt", "value": ""}
        ]
        
        # History management
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.past_key_values = None
        self.output_ids = None
        self.step_count = 0
        
        # Get image processor
        self.image_processor = model.get_vision_tower().image_processor
        
        # Initialize model state
        self.model.reset(1)
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def _preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        """Preprocess depth image to match model input size."""
        from transformers.image_utils import to_numpy_array
        
        target_height = self.image_processor.crop_size['height']  # 384
        target_width = self.image_processor.crop_size['width']   # 384
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
        
        return img, (target_width, target_height)
    
    def _get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        """Get camera intrinsic matrix from sensor config."""
        
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
    
    def _preprocess_intrinsic(self, intrinsic, ori_size, target_size):
        """Preprocess intrinsic matrix for resized image."""
        import copy
        
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height
        
        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def _get_axis_align_matrix(self):
        """Get axis alignment matrix."""
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
        return ma
    
    def _xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        """Convert xyz position and yaw to transformation matrix."""
        
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix
    
    def _parse_actions(self, output: str) -> list:
        """Parse action sequence from StreamVLN output."""
        import re
        
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        # actions2idx values are integers, not lists, so directly map them
        actions = [self.actions2idx[match] for match in matches]
        return actions
    
    def _preprocess_qwen(self, sources, has_image: bool = False, add_system: bool = False):
        """Preprocess input for Qwen model."""
        import copy
        from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX
        
        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(self.tokenizer)
        
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        unmask_tokens_idx = [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = "you can see " + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : "You are a helpful assistant."}])

            for conv in source:
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role = roles.get(role, role)
                conv = [{"role" : role, "content" : content}]
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id

            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids, conversations
    
    def predict_action(
        self,
        observation: Any,  # Can be Image or dict with observations
        instruction: str,
        plan: Optional[str] = None,
        prev_subtask: Optional[str] = None,
        **kwargs
    ) -> Tuple[int, Optional[str]]:
        """
        Predict action from observation and instruction.
        
        For StreamVLN, observation can be:
        - dict: Contains 'rgb', 'depth', 'gps', 'compass', 'env', 'sensor_config', etc.
        - Image: Fallback to single frame processing (not recommended)
        
        This method encapsulates all StreamVLN-specific logic:
        - Multi-modal data processing (RGB + Depth + Pose + Intrinsics)
        - Video sequence management
        - Action sequence generation
        """
        import copy
        try:
            import quaternion
        except ImportError:
            quaternion = None  # Will handle error if needed
        
        # Import StreamVLN specific utilities
        try:
            from depth_camera_filtering import filter_depth
        except ImportError:
            # Fallback if depth_camera_filtering not available
            def filter_depth(depth, blur_type=None):
                return depth
        
        try:
            from streamvln.utils.utils import DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN, dict_to_cuda
        except ImportError:
            # Fallback
            DEFAULT_MEMORY_TOKEN = "<memory>"
            DEFAULT_VIDEO_TOKEN = "<video>"
            def dict_to_cuda(d, device):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        
        # Handle observation input
        if isinstance(observation, dict):
            # Extract from observations dict
            rgb = observation.get('rgb')
            depth = observation.get('depth')
            gps = observation.get('gps', [0, 0])
            compass = observation.get('compass', [0])
            env = observation.get('env')
            sensor_config = observation.get('sensor_config')
            initial_height = observation.get('initial_height', 0)
            camera_height = observation.get('camera_height', 1.25)
            min_depth = observation.get('min_depth', 0.0)
            max_depth = observation.get('max_depth', 10.0)
        else:
            # Fallback: single frame (not ideal for StreamVLN)
            print("Warning: StreamVLN works best with full observations dict")
            return self.actions2idx['STOP'], None
        
        # If we have action sequence, return next action
        if len(self.action_seq) > 0:
            action = self.action_seq.pop(0)
            return action, None
        
        # Process depth
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        depth = depth * (max_depth - min_depth) + min_depth
        depth = depth * 1000
        
        # Get agent state and compute pose
        if env is not None:
            agent_state = env.sim.get_agent_state()
            height = agent_state.position[1] - initial_height
        else:
            height = 0
        
        x, y = gps[0], gps[1]
        camera_yaw = compass[0] if len(compass) > 0 else 0
        camera_position = np.array([x, -y, camera_height + height])
        tf_camera_to_episodic = self._xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
        
        # Process RGB image
        image = Image.fromarray(rgb).convert('RGB')
        image_size = image.size
        image_tensor = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
        
        # Process depth image
        depth_image, resize_shape = self._preprocess_depth_image(
            Image.fromarray(depth.astype(np.uint16), mode='I;16'), 
            do_depth_scale=True
        )
        
        # Process intrinsics
        if sensor_config is not None:
            intrinsic_matrix = self._get_intrinsic_matrix(sensor_config.rgb_sensor)
        else:
            # Default intrinsics
            intrinsic_matrix = np.eye(4)
        
        intrinsic = self._preprocess_intrinsic(intrinsic_matrix, image_size, resize_shape)
        intrinsic = torch.from_numpy(intrinsic).float()
        
        # Update history
        self.time_ids.append(self.step_count)
        self.rgb_list.append(image_tensor)
        self.depth_list.append(torch.from_numpy(depth_image).float())
        self.pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self._get_axis_align_matrix())
        self.intrinsic_list.append(intrinsic)
        
        # Prepare input for model
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace(
            ' Where should you go next to stay on track?',
            ' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.'
        )
        if self.step_count != 0:
            sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
        sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
        sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
        add_system = True
        
        input_ids, conversations = self._preprocess_qwen([sources], True, add_system=add_system)
        if self.output_ids is not None:
            input_ids = torch.cat([self.output_ids, input_ids.to(self.output_ids.device)], dim=1)
        
        # Select frames (current + history if needed)
        images = self.rgb_list[-1:]
        depths = self.depth_list[-1:]
        poses = self.pose_list[-1:]
        intrinsics = self.intrinsic_list[-1:]
        
        if self.step_count != 0 and self.step_count % self.num_frames == 0:
            if self.num_history is None:
                history_ids = slice(0, len(self.time_ids), self.num_future_steps)
            else:
                step_interval = max(1, len(self.time_ids) // self.num_history)
                history_ids = slice(0, len(self.time_ids), step_interval)
            
            # Select history frames
            history_indices = list(range(*history_ids.indices(len(self.rgb_list))))
            if history_indices:
                images = [self.rgb_list[i] for i in history_indices] + images
                depths = [self.depth_list[i] for i in history_indices] + depths
                poses = [self.pose_list[i] for i in history_indices] + poses
                intrinsics = [self.intrinsic_list[i] for i in history_indices] + intrinsics
        
        # Prepare input dict
        # Ensure time_ids is not empty (encode_rgbd expects at least one element)
        time_ids_for_model = self.time_ids if len(self.time_ids) > 0 else [0]
        
        input_dict = {
            'images': torch.stack(images).unsqueeze(0),
            'depths': torch.stack(depths).unsqueeze(0),
            'poses': torch.stack(poses).unsqueeze(0),
            'intrinsics': torch.stack(intrinsics).unsqueeze(0),
            'inputs': input_ids,
            'env_id': self.env_id,
            'time_ids': [time_ids_for_model],
            'task_type': [0]
        }
        
        input_dict = dict_to_cuda(input_dict, self.device)
        
        for key, value in input_dict.items():
            if key in ['images', 'depths', 'poses', 'intrinsics']:
                input_dict[key] = input_dict[key].to(torch.bfloat16)
        
        # Generate action sequence
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_dict,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=10000,
                    use_cache=True,
                    return_dict_in_generate=True,
                    past_key_values=self.past_key_values
                )
            
            # Check if outputs is valid
            if outputs is None:
                raise ValueError("Model generate returned None")
            
            # Check if sequences exist
            if not hasattr(outputs, 'sequences') or outputs.sequences is None:
                raise ValueError("Outputs.sequences is None")
            
            self.output_ids = outputs.sequences
            self.past_key_values = getattr(outputs, 'past_key_values', None)
            
            # Check if output_ids is valid before decoding
            if self.output_ids is None or self.output_ids.numel() == 0:
                raise ValueError("output_ids is None or empty")
            
            decoded_outputs = self.tokenizer.batch_decode(self.output_ids, skip_special_tokens=False)
            if not decoded_outputs or len(decoded_outputs) == 0:
                raise ValueError("batch_decode returned empty list")
            
            llm_outputs = decoded_outputs[0].strip()
            
            self.action_seq = self._parse_actions(llm_outputs)
            if len(self.action_seq) == 0:
                self.action_seq = [0]  # Default to stop
            
        except Exception as e:
            self.action_seq = [0]
            self.output_ids = None
            self.past_key_values = None
        
        # Reset if needed (after processing current step)
        if self.step_count % self.num_frames == 0:
            self.model.reset_for_env(self.env_id)
            self.output_ids = None
            self.past_key_values = None
            # Clear history lists to prevent memory growth
            self.rgb_list = []
            self.depth_list = []
            self.pose_list = []
            self.intrinsic_list = []
            self.time_ids = []
        
        self.step_count += 1
        
        # Return first action from sequence
        if len(self.action_seq) > 0:
            action = self.action_seq.pop(0)
            return action, None
        else:
            return self.actions2idx['STOP'], None
