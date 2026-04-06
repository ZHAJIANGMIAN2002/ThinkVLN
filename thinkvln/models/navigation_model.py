#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Navigation Model Interface

This module defines a unified interface for navigation models.
All navigation models should implement this interface to work with the evaluator.
"""

from abc import ABC, abstractmethod
import math
from typing import Optional, Tuple, Any, List, Dict
from PIL import Image
import torch
import numpy as np

from thinkvln.tools.dataset_utils import select_memory_frame_indices


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

    def reset_episode_state(self, episode_key: Optional[str] = None):
        self.episode_key = episode_key
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.past_key_values = None
        self.output_ids = None
        self.step_count = 0
        self.last_subtask_id = None
        self.last_subgoal = None
        self.prev_progress = 0.0
        self._last_predicted_progress = 0.0
        self._last_predicted_done = False
        self._last_debug_snapshot: Optional[Dict[str, Any]] = None
        if hasattr(self.model, "reset_for_env") and hasattr(self, "env_id"):
            self.model.reset_for_env(self.env_id)

    def get_last_debug_snapshot(self) -> Optional[Dict[str, Any]]:
        if self._last_debug_snapshot is None:
            return None
        return dict(self._last_debug_snapshot)

    @staticmethod
    def _augment_instruction(instruction: str, subgoal: Optional[str] = None, hint: Optional[str] = None) -> str:
        text = f"Instruction: {str(instruction or '').strip()}"
        if str(subgoal or "").strip():
            text = f"{text}\nCurrent subtask: {str(subgoal).strip()}"
        if str(hint or "").strip():
            text = f"{text}\nWatcher hint: {str(hint).strip()}"
        return text

    def _infer_progress_done_from_aux_head(
        self,
        input_dict: Dict[str, Any],
        fallback_action: int,
    ) -> Tuple[float, bool]:
        if not hasattr(self.model, "predict_progress_done"):
            progress = 1.0 if int(fallback_action) == self.actions2idx["STOP"] else float(self.prev_progress)
            done = bool(int(fallback_action) == self.actions2idx["STOP"] or progress > self.done_threshold)
            return progress, done
        try:
            progress_preds, done_preds = self.model.predict_progress_done(
                input_ids=input_dict["inputs"],
                images=input_dict["images"],
                depths=input_dict["depths"],
                poses=input_dict["poses"],
                intrinsics=input_dict["intrinsics"],
                time_ids=input_dict.get("time_ids"),
                task_type=input_dict.get("task_type"),
            )
            progress = float(progress_preds[0].detach().float().cpu().item())
            done_prob = float(done_preds[0].detach().float().cpu().item())
            progress = max(0.0, min(1.0, progress))
            done = bool(done_prob > 0.5)
            return progress, done
        except Exception:
            progress = 1.0 if int(fallback_action) == self.actions2idx["STOP"] else float(self.prev_progress)
            done = bool(int(fallback_action) == self.actions2idx["STOP"] or progress > self.done_threshold)
            return progress, done
    
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


class ThinkVLNActorNavigationModel(NavigationModel):
    """ThinkVLNActor wrapper for closed-loop action/progress inference."""

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        memory_num_history_images: int = 8,
        done_threshold: float = 0.85,
        use_memory: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.device = device

        self.actions2idx = {
            "stop": 0,
            "forward": 1,
            "turn_left": 2,
            "turn_right": 3,
        }
        self.num_query_tokens = int(getattr(model.config, "num_query_tokens", 4))
        self.action_query_token_id = int(getattr(model.config, "action_query_token_id", 151700))
        self.progress_query_token_id = int(getattr(model.config, "progress_query_token_id", 151701))
        self.memory_num_history_images = int(memory_num_history_images)
        self.done_threshold = float(done_threshold)
        self.use_memory = bool(use_memory)
        self.prompt_template = (
            "Instruction: {instruction}\n"
            "Current subgoal: {subgoal}\n"
            "Previous progress: {prev_progress:.3f}\n"
            "{memory_hint}"
            "Predict the next 4 actions and the current-step subgoal progress."
        )
        self.reset_episode_state()

    def eval(self):
        self.model.eval()

    def reset_episode_state(self, episode_key: Optional[str] = None):
        self.episode_key = episode_key
        self.memory_bank_images: List[Image.Image] = []
        self.memory_bank_subtasks: List[int] = []
        self.subtask_start_bank_indices: List[int] = []
        self.current_subtask_id = 1
        self.last_subtask_id = None
        self.last_subgoal = None
        self.prev_progress = 0.0
        self._last_debug_snapshot: Optional[Dict[str, Any]] = None

    def _update_subtask_state(
        self,
        subgoal: str,
        subtask_id: Optional[int] = None,
    ) -> Tuple[int, bool]:
        is_subtask_start = False
        if subtask_id is not None:
            resolved_subtask_id = max(1, int(subtask_id))
            if self.last_subtask_id is None:
                is_subtask_start = True
            elif resolved_subtask_id != self.last_subtask_id:
                is_subtask_start = True
                self.prev_progress = 0.0
            self.current_subtask_id = resolved_subtask_id
            self.last_subtask_id = resolved_subtask_id
            self.last_subgoal = subgoal
            return resolved_subtask_id, is_subtask_start

        if self.last_subgoal is None:
            self.last_subgoal = subgoal
            self.last_subtask_id = self.current_subtask_id
            return self.current_subtask_id, True
        if subgoal != self.last_subgoal:
            self.current_subtask_id += 1
            self.last_subgoal = subgoal
            self.last_subtask_id = self.current_subtask_id
            self.prev_progress = 0.0
            is_subtask_start = True
        else:
            self.last_subtask_id = self.current_subtask_id
        return self.current_subtask_id, is_subtask_start

    def _build_query_token_id_list(self) -> List[int]:
        tokens: List[int] = []
        for _ in range(self.num_query_tokens):
            tokens.append(self.action_query_token_id)
            tokens.append(self.progress_query_token_id)
        return tokens

    def _build_query_tokens(self) -> torch.Tensor:
        tokens = self._build_query_token_id_list()
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

    def _build_inputs(self, images: List[Image.Image], prompt: str) -> dict:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{
            "role": "user",
            "content": content,
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=False,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        query_tokens = self._build_query_tokens()
        input_ids = torch.cat([input_ids, query_tokens], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, query_tokens.shape[1]), dtype=torch.long, device=self.device)],
            dim=1,
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_labels": torch.zeros(
                (1, self.num_query_tokens),
                dtype=torch.long,
                device=self.device,
            ),
            "progress_labels": torch.zeros((1,), dtype=torch.float32, device=self.device),
            "done_labels": torch.zeros((1,), dtype=torch.float32, device=self.device),
        }
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            model_inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            model_inputs["image_grid_thw"] = inputs["image_grid_thw"].to(self.device)
        return model_inputs

    @staticmethod
    def _first_scalar(x: Optional[torch.Tensor], default: float = 0.0) -> float:
        if x is None:
            return float(default)
        if x.ndim == 0:
            return float(x.item())
        if x.ndim == 1:
            return float(x[0].item())
        return float(x[0, 0].item())

    @staticmethod
    def _first_vector_logits(action_logits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if action_logits is None:
            return None
        if action_logits.ndim == 2:
            return action_logits[0]
        if action_logits.ndim == 3:
            return action_logits[0, 0]
        return None

    @staticmethod
    def select_action_from_logits(
        action_logits: Optional[torch.Tensor],
        sample_action: bool = False,
        action_generator: Optional[torch.Generator] = None,
        forbidden_actions: Optional[List[int]] = None,
    ) -> int:
        if action_logits is None:
            return 0

        logits = action_logits.detach().to(device="cpu", dtype=torch.float32)
        if logits.ndim != 1 or logits.numel() == 0:
            return 0
        if forbidden_actions:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)
            for action_idx in forbidden_actions:
                if 0 <= int(action_idx) < logits.numel():
                    valid_mask[int(action_idx)] = False
            if not bool(valid_mask.any().item()):
                return 0
            logits = logits.clone()
            logits[~valid_mask] = float("-inf")
        if not sample_action:
            return int(torch.argmax(logits, dim=-1).item())

        probs = torch.softmax(logits, dim=-1)
        if (not torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())
        sampled = torch.multinomial(probs, 1, generator=action_generator)
        return int(sampled.item())

    def _maybe_reset_for_episode(self, episode_key: Optional[str]):
        if episode_key is not None and episode_key != self.episode_key:
            self.reset_episode_state(episode_key=episode_key)

    def _append_observation_to_memory_bank(self, observation: Image.Image, subtask_id: int, is_subtask_start: bool) -> None:
        if not self.use_memory:
            return
        self.memory_bank_images.append(observation)
        self.memory_bank_subtasks.append(subtask_id)
        frame_idx = len(self.memory_bank_images) - 1
        if is_subtask_start or frame_idx == 0:
            self.subtask_start_bank_indices.append(frame_idx)

    def record_memory_observation(
        self,
        observation: Image.Image,
        subtask_id: int,
        episode_key: Optional[str] = None,
    ) -> None:
        self._maybe_reset_for_episode(episode_key)
        try:
            resolved_subtask_id = int(subtask_id)
        except (TypeError, ValueError):
            resolved_subtask_id = 1
        resolved_subtask_id = max(1, resolved_subtask_id)
        is_subtask_start = self.last_subtask_id is None or resolved_subtask_id != self.last_subtask_id
        self.current_subtask_id = resolved_subtask_id
        self.last_subtask_id = resolved_subtask_id
        self._append_observation_to_memory_bank(
            observation=observation,
            subtask_id=resolved_subtask_id,
            is_subtask_start=is_subtask_start,
        )

    def _prepare_memory_images_from_bank(self) -> Tuple[List[Image.Image], List[int], List[int]]:
        if not self.use_memory:
            return [], [], []
        if not self.memory_bank_images:
            return [], [], []
        frame_idx = len(self.memory_bank_images) - 1
        subtask_sequence = self.memory_bank_subtasks
        history_indices = select_memory_frame_indices(
            frame_idx=frame_idx,
            subtask_sequence=subtask_sequence,
            memory_num_history_images=self.memory_num_history_images,
        )
        selected_indices = history_indices + [frame_idx]
        images = [
            self.memory_bank_images[idx]
            for idx in selected_indices
            if 0 <= idx < len(self.memory_bank_images)
        ]
        selected_subtask_ids = [
            int(self.memory_bank_subtasks[idx])
            for idx in selected_indices
            if 0 <= idx < len(self.memory_bank_subtasks)
        ]
        return images, list(selected_indices), selected_subtask_ids

    def get_last_debug_snapshot(self) -> Optional[Dict[str, Any]]:
        if self._last_debug_snapshot is None:
            return None
        snapshot = dict(self._last_debug_snapshot)
        for key in (
            "query_token_ids",
            "selected_indices",
            "selected_subtask_ids",
            "memory_bank_subtasks",
            "selected_images",
        ):
            value = snapshot.get(key)
            if isinstance(value, list):
                snapshot[key] = list(value)
        return snapshot

    def predict_action_with_progress_and_done(
        self,
        observation: Image.Image,
        instruction: str,
        subgoal: str,
        episode_key: Optional[str] = None,
        subtask_id: Optional[int] = None,
        hint: Optional[str] = None,
        sample_action: bool = False,
        action_generator: Optional[torch.Generator] = None,
        forbidden_actions: Optional[List[int]] = None,
    ) -> Tuple[int, float, bool]:
        self._maybe_reset_for_episode(episode_key)
        resolved_subtask_id, is_subtask_start = self._update_subtask_state(subgoal, subtask_id=subtask_id)
        prev_progress_input = float(self.prev_progress)
        if self.use_memory:
            self._append_observation_to_memory_bank(
                observation=observation,
                subtask_id=resolved_subtask_id,
                is_subtask_start=is_subtask_start,
            )
            images, selected_indices, selected_subtask_ids = self._prepare_memory_images_from_bank()
        else:
            images = [observation]
            selected_indices = [0]
            selected_subtask_ids = [resolved_subtask_id]
        memory_hint = (
            "Historical observations are provided.\n"
            if len(images) > 1
            else ""
        )
        watcher_hint = (
            f"Hint from watcher: {str(hint).strip()}\n"
            if str(hint or "").strip()
            else ""
        )
        prompt = self.prompt_template.format(
            instruction=instruction,
            subgoal=subgoal,
            prev_progress=prev_progress_input,
            memory_hint=memory_hint + watcher_hint,
        )
        query_token_ids = self._build_query_token_id_list()
        self._last_debug_snapshot = {
            "prompt": prompt,
            "query_token_ids": query_token_ids,
            "selected_indices": list(selected_indices),
            "selected_subtask_ids": list(selected_subtask_ids),
            "prev_progress_input": prev_progress_input,
            "predicted_progress": None,
            "memory_bank_size": len(self.memory_bank_images),
            "memory_bank_subtasks": list(self.memory_bank_subtasks),
            "selected_images": list(images),
        }
        model_inputs = self._build_inputs(images, prompt)

        with torch.no_grad():
            outputs = self.model(**model_inputs, return_dict=True)

        action_logits = outputs.get("action_logits")
        first_step_logits = self._first_vector_logits(action_logits)
        action = self.select_action_from_logits(
            first_step_logits,
            sample_action=sample_action,
            action_generator=action_generator,
            forbidden_actions=forbidden_actions,
        )

        progress_preds = outputs.get("progress_preds")
        progress = max(0.0, min(1.0, self._first_scalar(progress_preds, default=0.0)))
        if self._last_debug_snapshot is not None:
            self._last_debug_snapshot["predicted_progress"] = float(progress)

        done_probs = outputs.get("done_preds")
        if done_probs is not None:
            done_prob = self._first_scalar(done_probs, default=0.0)
        else:
            done_logits = outputs.get("done_logits")
            if done_logits is not None:
                done_prob = self._first_scalar(torch.sigmoid(done_logits), default=0.0)
            else:
                done_prob = 1.0 if progress > self.done_threshold else 0.0
        done = bool(done_prob > 0.5)

        self.prev_progress = progress
        return action, progress, done

    def predict_action_with_progress(
        self,
        observation: Image.Image,
        instruction: str,
        subgoal: str,
        episode_key: Optional[str] = None,
        subtask_id: Optional[int] = None,
        hint: Optional[str] = None,
    ) -> Tuple[int, float]:
        action, progress, _ = self.predict_action_with_progress_and_done(
            observation=observation,
            instruction=instruction,
            subgoal=subgoal,
            episode_key=episode_key,
            subtask_id=subtask_id,
            hint=hint,
        )
        return action, progress

    def predict_action(
        self,
        observation: Image.Image,
        instruction: str,
        plan: Optional[str] = None,
        prev_subtask: Optional[str] = None,
        **kwargs
    ) -> Tuple[int, Optional[str]]:
        subgoal = kwargs.get("subgoal") or plan or "Navigate to the goal."
        try:
            action, _ = self.predict_action_with_progress(
                observation=observation,
                instruction=instruction,
                subgoal=subgoal,
                episode_key=kwargs.get("episode_key"),
                subtask_id=kwargs.get("subtask_id"),
                hint=kwargs.get("hint"),
            )
            return action, None
        except Exception:
            return self.actions2idx["stop"], None


class ThinkVLNFMNavigationModel(ThinkVLNActorNavigationModel):
    """Navigation wrapper for flow-matching waypoint actor."""

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        memory_num_history_images: int = 8,
        done_threshold: float = 0.85,
        use_memory: bool = True,
        stop_distance_threshold: float = 0.06,
        turn_ratio_threshold: float = 0.6,
    ):
        super().__init__(
            model=model,
            processor=processor,
            device=device,
            memory_num_history_images=memory_num_history_images,
            done_threshold=done_threshold,
            use_memory=use_memory,
        )
        self.stop_distance_threshold = float(stop_distance_threshold)
        self.turn_ratio_threshold = float(turn_ratio_threshold)
        self.prompt_template = (
            "Instruction: {instruction}\n"
            "Current subgoal: {subgoal}\n"
            "Previous progress: {prev_progress:.3f}\n"
            "{memory_hint}"
            "Predict waypoint deltas for the next horizon."
        )

    def _build_inputs(self, images: List[Image.Image], prompt: str) -> dict:
        model_inputs = super()._build_inputs(images, prompt)
        model_inputs.pop("action_labels", None)
        model_inputs.pop("progress_labels", None)
        model_inputs.pop("done_labels", None)
        return model_inputs

    def _waypoint_to_action_logits(self, waypoint_preds: Optional[torch.Tensor]) -> torch.Tensor:
        logits = torch.full((4,), -2.0, dtype=torch.float32)
        if waypoint_preds is None or waypoint_preds.numel() == 0:
            logits[0] = 2.0
            return logits

        if waypoint_preds.ndim == 3:
            first_wp = waypoint_preds[0, 0]
        elif waypoint_preds.ndim == 2:
            first_wp = waypoint_preds[0]
        else:
            logits[0] = 2.0
            return logits

        dx = float(first_wp[0].item()) if first_wp.numel() > 0 else 0.0
        dz = float(first_wp[1].item()) if first_wp.numel() > 1 else 0.0
        dist = math.sqrt(dx * dx + dz * dz)
        lateral_ratio = abs(dx) / max(abs(dz), 1e-6)

        if dist <= self.stop_distance_threshold:
            logits[0] = 3.0
            return logits

        if lateral_ratio >= self.turn_ratio_threshold:
            if dx >= 0:
                logits[3] = 2.5
                logits[2] = -0.5
            else:
                logits[2] = 2.5
                logits[3] = -0.5
            logits[1] = 0.3
        else:
            logits[1] = 2.5
            logits[2] = 0.1
            logits[3] = 0.1
        return logits

    def predict_action_with_progress_and_done(
        self,
        observation: Image.Image,
        instruction: str,
        subgoal: str,
        episode_key: Optional[str] = None,
        subtask_id: Optional[int] = None,
        hint: Optional[str] = None,
        sample_action: bool = False,
        action_generator: Optional[torch.Generator] = None,
        forbidden_actions: Optional[List[int]] = None,
    ) -> Tuple[int, float, bool]:
        self._maybe_reset_for_episode(episode_key)
        resolved_subtask_id, is_subtask_start = self._update_subtask_state(subgoal, subtask_id=subtask_id)
        prev_progress_input = float(self.prev_progress)

        if self.use_memory:
            self._append_observation_to_memory_bank(
                observation=observation,
                subtask_id=resolved_subtask_id,
                is_subtask_start=is_subtask_start,
            )
            images, selected_indices, selected_subtask_ids = self._prepare_memory_images_from_bank()
        else:
            images = [observation]
            selected_indices = [0]
            selected_subtask_ids = [resolved_subtask_id]

        memory_hint = "Historical observations are provided.\n" if len(images) > 1 else ""
        watcher_hint = (
            f"Hint from watcher: {str(hint).strip()}\n"
            if str(hint or "").strip()
            else ""
        )
        prompt = self.prompt_template.format(
            instruction=instruction,
            subgoal=subgoal,
            prev_progress=prev_progress_input,
            memory_hint=memory_hint + watcher_hint,
        )
        query_token_ids = self._build_query_token_id_list()
        self._last_debug_snapshot = {
            "prompt": prompt,
            "query_token_ids": query_token_ids,
            "selected_indices": list(selected_indices),
            "selected_subtask_ids": list(selected_subtask_ids),
            "prev_progress_input": prev_progress_input,
            "predicted_progress": None,
            "memory_bank_size": len(self.memory_bank_images),
            "memory_bank_subtasks": list(self.memory_bank_subtasks),
            "selected_images": list(images),
            "waypoint_first": None,
        }

        model_inputs = self._build_inputs(images, prompt)
        with torch.no_grad():
            outputs = self.model(**model_inputs, return_dict=True)

        waypoint_preds = outputs.get("waypoint_preds")
        action_logits = self._waypoint_to_action_logits(waypoint_preds)
        action = self.select_action_from_logits(
            action_logits,
            sample_action=sample_action,
            action_generator=action_generator,
            forbidden_actions=forbidden_actions,
        )

        if waypoint_preds is not None and waypoint_preds.numel() > 0:
            first = waypoint_preds[0, 0] if waypoint_preds.ndim == 3 else waypoint_preds[0]
            dx = float(first[0].item()) if first.numel() > 0 else 0.0
            dz = float(first[1].item()) if first.numel() > 1 else 0.0
            dist = math.sqrt(dx * dx + dz * dz)
            if self._last_debug_snapshot is not None:
                self._last_debug_snapshot["waypoint_first"] = [dx, dz]
        else:
            dist = 0.0

        progress_increment = min(0.25, dist / 0.5)
        if action == self.actions2idx["stop"]:
            progress_increment = max(progress_increment, 0.05)
        progress = max(0.0, min(1.0, prev_progress_input + progress_increment))

        if self._last_debug_snapshot is not None:
            self._last_debug_snapshot["predicted_progress"] = float(progress)

        done = bool(action == self.actions2idx["stop"] or progress > self.done_threshold)
        self.prev_progress = progress
        return action, progress, done


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
        env_id: int = 0,
        done_threshold: float = 0.85,
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
        self.done_threshold = float(done_threshold)
        
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
        
        # Get image processor
        self.image_processor = model.get_vision_tower().image_processor
        self.model.reset(1)
        self.reset_episode_state()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def reset_episode_state(self, episode_key: Optional[str] = None):
        self.episode_key = episode_key
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.past_key_values = None
        self.output_ids = None
        self.step_count = 0
        self.last_subtask_id = None
        self.last_subgoal = None
        self.prev_progress = 0.0
        self._last_predicted_progress = 0.0
        self._last_predicted_done = False
        self._last_debug_snapshot: Optional[Dict[str, Any]] = None
        self.memory_bank_images = []
        self.memory_bank_subtasks = []
        if hasattr(self.model, "reset_for_env"):
            self.model.reset_for_env(self.env_id)

    def get_last_debug_snapshot(self) -> Optional[Dict[str, Any]]:
        if self._last_debug_snapshot is None:
            return None
        return dict(self._last_debug_snapshot)

    @staticmethod
    def _augment_instruction(
        instruction: str,
        subgoal: Optional[str] = None,
        hint: Optional[str] = None,
        previous_progress: Optional[float] = None,
    ) -> str:
        text = f"Instruction: {str(instruction or '').strip()}"
        if str(subgoal or "").strip():
            text = f"{text}\nCurrent subtask: {str(subgoal).strip()}"
        if str(hint or "").strip():
            text = f"{text}\nWatcher hint: {str(hint).strip()}"
        if previous_progress is not None:
            text = f"{text}\nPrevious progress: {float(previous_progress):.4f}"
        return text

    def record_memory_observation(
        self,
        observation: Any,
        subtask_id: Optional[int] = None,
        episode_key: Optional[str] = None,
    ) -> None:
        del observation
        if episode_key is not None and episode_key != self.episode_key:
            self.reset_episode_state(episode_key=episode_key)
        self.memory_bank_images.append(None)
        self.memory_bank_subtasks.append(int(subtask_id) if subtask_id is not None else 0)

    def _infer_progress_done_from_aux_head(
        self,
        input_dict: Dict[str, Any],
        fallback_action: int,
    ) -> Tuple[float, bool]:
        if not hasattr(self.model, "predict_progress_done"):
            progress = 1.0 if int(fallback_action) == self.actions2idx["STOP"] else float(self.prev_progress)
            done = bool(int(fallback_action) == self.actions2idx["STOP"] or progress > self.done_threshold)
            return progress, done
        try:
            progress_preds, done_preds = self.model.predict_progress_done(
                input_ids=input_dict["inputs"],
                images=input_dict["images"],
                depths=input_dict["depths"],
                poses=input_dict["poses"],
                intrinsics=input_dict["intrinsics"],
                time_ids=input_dict.get("time_ids"),
                task_type=input_dict.get("task_type"),
            )
            progress = float(progress_preds[0].detach().float().cpu().item())
            done_prob = float(done_preds[0].detach().float().cpu().item())
            progress = max(0.0, min(1.0, progress))
            done = bool(done_prob > 0.5)
            return progress, done
        except Exception:
            progress = 1.0 if int(fallback_action) == self.actions2idx["STOP"] else float(self.prev_progress)
            done = bool(int(fallback_action) == self.actions2idx["STOP"] or progress > self.done_threshold)
            return progress, done
    
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

        if kwargs.get("episode_key") is not None and kwargs.get("episode_key") != self.episode_key:
            self.reset_episode_state(episode_key=kwargs.get("episode_key"))

        subgoal = kwargs.get("subgoal")
        hint = kwargs.get("hint")
        need_progress_done = bool(kwargs.get("need_progress_done", False))
        instruction_text = self._augment_instruction(
            instruction,
            subgoal=subgoal,
            hint=hint,
            previous_progress=float(self.prev_progress),
        )
        
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
            rgb = np.array(observation, copy=False) if not isinstance(observation, Image.Image) else np.array(observation)
            depth = None
            gps = [0.0, 0.0]
            compass = [0.0]
            env = None
            sensor_config = None
            initial_height = 0.0
            camera_height = 1.25
            min_depth = 0.0
            max_depth = 10.0

        if rgb is None:
            return self.actions2idx['STOP'], None
        rgb = np.asarray(rgb)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            return self.actions2idx['STOP'], None

        if depth is None:
            depth = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
        else:
            depth = np.asarray(depth)
            if depth.ndim == 2:
                depth = depth[:, :, None]
            if depth.ndim != 3:
                depth = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)

        if need_progress_done and len(self.action_seq) > 0:
            # Keep progress/done aligned with each control step by forcing a fresh forward.
            self.action_seq = []

        # If we have action sequence, return next action
        if len(self.action_seq) > 0:
            action = self.action_seq.pop(0)
            if need_progress_done:
                done = bool(action == self.actions2idx["STOP"] or self._last_predicted_done)
                self._last_predicted_done = done
            self._last_debug_snapshot = {
                "instruction_text": instruction_text,
                "predicted_progress": float(self._last_predicted_progress),
                "predicted_done": bool(self._last_predicted_done),
                "used_cached_action_seq": True,
                "returned_action": int(action),
            }
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
        sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction_text)
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

        if need_progress_done:
            fallback_action = self.action_seq[0] if self.action_seq else self.actions2idx["STOP"]
            progress, done = self._infer_progress_done_from_aux_head(
                input_dict=input_dict,
                fallback_action=int(fallback_action),
            )
            self._last_predicted_progress = progress
            self._last_predicted_done = done
        
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
            if need_progress_done:
                # Do not reuse stale aux predictions across cached chunk steps.
                self.action_seq = []
            self._last_debug_snapshot = {
                "instruction_text": instruction_text,
                "predicted_progress": float(self._last_predicted_progress),
                "predicted_done": bool(self._last_predicted_done),
                "used_cached_action_seq": False,
                "returned_action": int(action),
            }
            return action, None
        else:
            self._last_debug_snapshot = {
                "instruction_text": instruction_text,
                "predicted_progress": float(self._last_predicted_progress),
                "predicted_done": bool(self._last_predicted_done),
                "used_cached_action_seq": False,
                "returned_action": int(self.actions2idx['STOP']),
            }
            return self.actions2idx['STOP'], None

    def predict_action_with_progress_and_done(
        self,
        observation: Any,
        instruction: str,
        subgoal: str,
        episode_key: Optional[str] = None,
        subtask_id: Optional[int] = None,
        hint: Optional[str] = None,
        sample_action: bool = False,
        action_generator: Optional[torch.Generator] = None,
        forbidden_actions: Optional[List[int]] = None,
    ) -> Tuple[int, float, bool]:
        del sample_action, action_generator
        if episode_key is not None and episode_key != self.episode_key:
            self.reset_episode_state(episode_key=episode_key)

        if subtask_id is not None:
            resolved_subtask_id = max(1, int(subtask_id))
            if self.last_subtask_id is None or resolved_subtask_id != self.last_subtask_id:
                self.prev_progress = 0.0
            self.last_subtask_id = resolved_subtask_id
        else:
            if self.last_subgoal is None or str(subgoal) != str(self.last_subgoal):
                self.prev_progress = 0.0
        self.last_subgoal = str(subgoal)

        action, _ = self.predict_action(
            observation=observation,
            instruction=instruction,
            subgoal=subgoal,
            hint=hint,
            episode_key=episode_key,
            need_progress_done=True,
        )

        progress = float(self._last_predicted_progress)
        done = bool(self._last_predicted_done or int(action) == self.actions2idx["STOP"])
        if forbidden_actions and int(action) in [int(x) for x in forbidden_actions]:
            action = self.actions2idx["STOP"]
            done = True
            progress = max(progress, self.done_threshold)

        progress = max(0.0, min(1.0, float(progress)))
        self.prev_progress = progress
        if self._last_debug_snapshot is not None:
            self._last_debug_snapshot["predicted_progress"] = progress
            self._last_debug_snapshot["predicted_done"] = done
            self._last_debug_snapshot["watcher_subgoal"] = str(subgoal)
            self._last_debug_snapshot["watcher_hint"] = str(hint or "")
        return int(action), progress, bool(done)

    def predict_action_with_progress(
        self,
        observation: Any,
        instruction: str,
        subgoal: str,
        episode_key: Optional[str] = None,
        subtask_id: Optional[int] = None,
        hint: Optional[str] = None,
    ) -> Tuple[int, float]:
        action, progress, _ = self.predict_action_with_progress_and_done(
            observation=observation,
            instruction=instruction,
            subgoal=subgoal,
            episode_key=episode_key,
            subtask_id=subtask_id,
            hint=hint,
        )
        return action, progress
