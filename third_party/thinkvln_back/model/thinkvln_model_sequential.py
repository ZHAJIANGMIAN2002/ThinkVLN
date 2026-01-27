import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from transformers import Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast


class ThinkVLNModel(Qwen3VLForConditionalGeneration):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Qwen3VL has hidden_size in text_config
        if hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'hidden_size'):
            hidden_size = self.config.text_config.hidden_size
        elif hasattr(self.config, 'hidden_size'):
            hidden_size = self.config.hidden_size
        else:
            raise AttributeError("Cannot find hidden_size in config")
        
        self.action_head = nn.Linear(hidden_size, 4, bias=False)
        
        self.action_loss_fn = nn.CrossEntropyLoss()
        
        self.action_token_id = getattr(self.config, 'action_token_id', None)
        
    def set_action_token_id(self, token_id: int):
        """Set the action token ID for identifying action positions in sequences."""
        self.action_token_id = token_id
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        action_labels: Optional[torch.LongTensor] = None,
        action_token_id: Optional[int] = None,
        action_loss_weight: Optional[float] = 1.0,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        current_action_token_id = action_token_id if action_token_id is not None else self.action_token_id
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if output_hidden_states is None:
            output_hidden_states = True
        
        # Get model outputs (hidden states)
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get hidden states
        hidden_states = model_outputs[0] if not return_dict else model_outputs.last_hidden_state
        
        # Compute language model logits
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        # Compute language modeling loss
        lm_loss = None
        if labels is not None:
            # Get vocab_size from config
            if hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'vocab_size'):
                vocab_size = self.config.text_config.vocab_size
            elif hasattr(self.config, 'vocab_size'):
                vocab_size = self.config.vocab_size
            else:
                vocab_size = logits.shape[-1]  # Fallback to logits dimension
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)
        
        # Compute action classification
        action_logits = None
        action_loss = None
        
        # Debug print (only for first call)
        if not hasattr(self, '_debug_printed'):
            print(f"\n[Model Debug] Forward conditions:")
            print(f"  current_action_token_id: {current_action_token_id}")
            print(f"  action_labels is not None: {action_labels is not None}")
            print(f"  input_ids is not None: {input_ids is not None}")
            if action_labels is not None:
                print(f"  action_labels shape: {action_labels.shape}")
                print(f"  action_labels: {action_labels}")
            if input_ids is not None:
                print(f"  input_ids shape: {input_ids.shape}")
                if current_action_token_id is not None:
                    has_token = (input_ids == current_action_token_id).any().item()
                    print(f"  Has action token in input_ids: {has_token}")
            self._debug_printed = True
        
        if current_action_token_id is not None and action_labels is not None and input_ids is not None:
            # Check hidden_states is valid
            if hidden_states is None:
                if not hasattr(self, '_debug_hidden_none'):
                    print(f"\n[Model Debug] hidden_states is None!")
                    self._debug_hidden_none = True
            else:
                # Find positions where action_token_id appears
                action_token_positions = (input_ids == current_action_token_id)
                
                # Extract hidden states at action token positions
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                # Debug: check dimensions match
                if not hasattr(self, '_debug_dim_check'):
                    print(f"\n[Model Debug] Dimension check:")
                    print(f"  input_ids shape: {input_ids.shape}")
                    print(f"  hidden_states shape: {hidden_states.shape}")
                    print(f"  action_token_positions shape: {action_token_positions.shape}")
                    self._debug_dim_check = True
                
                # For each sample in batch, find the action token position
                action_hidden_states = []
                valid_action_labels = []
                
                for batch_idx in range(batch_size):
                    # Check if this sample has valid action label
                    if action_labels[batch_idx] < 0:
                        if not hasattr(self, '_debug_skip_invalid'):
                            print(f"\n[Model Debug] Skipping batch_idx {batch_idx}: invalid label {action_labels[batch_idx]}")
                            self._debug_skip_invalid = True
                        continue
                    
                    # Find the position of action token in this sequence
                    action_positions = torch.where(action_token_positions[batch_idx])[0]
                    
                    if len(action_positions) > 0:
                        # Use the last occurrence of action token (most recent)
                        action_pos = action_positions[-1].item()
                        
                        # Debug: print for all samples in first batch
                        if not hasattr(self, '_debug_first_batch'):
                            print(f"\n[Model Debug] Processing batch_idx {batch_idx}:")
                            print(f"  action_pos: {action_pos}")
                            print(f"  hidden_states.shape: {hidden_states.shape}")
                            print(f"  action_pos < hidden_states.shape[1]: {action_pos < hidden_states.shape[1]}")
                            if batch_idx == batch_size - 1:
                                self._debug_first_batch = True
                        
                        # Ensure action_pos is within bounds
                        if action_pos < hidden_states.shape[1]:
                            # Extract hidden state at this position
                            try:
                                action_hidden = hidden_states[batch_idx, action_pos, :]
                                action_hidden_states.append(action_hidden)
                                valid_action_labels.append(action_labels[batch_idx])
                                
                                # Debug: print for all samples
                                if not hasattr(self, '_debug_extracted_all'):
                                    print(f"  [Model Debug] Successfully extracted batch_idx {batch_idx}! action_hidden.shape: {action_hidden.shape}")
                                    print(f"  action_hidden_states length: {len(action_hidden_states)}")
                                    if batch_idx == batch_size - 1:
                                        self._debug_extracted_all = True
                            except Exception as e:
                                import traceback
                                if not hasattr(self, '_debug_extract_error'):
                                    print(f"\n[Model Debug] Error extracting hidden state!")
                                    print(f"  batch_idx: {batch_idx}, action_pos: {action_pos}")
                                    print(f"  hidden_states.shape: {hidden_states.shape}")
                                    print(f"  hidden_states.device: {hidden_states.device}")
                                    print(f"  Error: {e}")
                                    print(f"  Traceback: {traceback.format_exc()}")
                                    self._debug_extract_error = True
                        else:
                            # Debug: position out of bounds
                            if not hasattr(self, '_debug_pos_oob'):
                                print(f"\n[Model Debug] Action pos out of bounds!")
                                print(f"  batch_idx: {batch_idx}, action_pos: {action_pos}, hidden_states.shape[1]: {hidden_states.shape[1]}")
                                self._debug_pos_oob = True
                    else:
                        # Debug: no action positions found
                        if not hasattr(self, '_debug_no_pos'):
                            print(f"\n[Model Debug] No action positions found for batch_idx {batch_idx}!")
                            print(f"  action_token_positions[batch_idx]: {action_token_positions[batch_idx].sum().item()} matches")
                            self._debug_no_pos = True
                
                # Debug: print summary after loop
                if not hasattr(self, '_debug_loop_summary'):
                    print(f"\n[Model Debug] Loop summary:")
                    print(f"  batch_size: {batch_size}")
                    print(f"  action_hidden_states collected: {len(action_hidden_states)}")
                    print(f"  valid_action_labels collected: {len(valid_action_labels)}")
                    for batch_idx in range(batch_size):
                        if action_labels[batch_idx] >= 0:
                            positions = torch.where(action_token_positions[batch_idx])[0]
                            print(f"  Sample {batch_idx}: action_label={action_labels[batch_idx]}, positions={positions.tolist()}")
                            if len(positions) > 0:
                                pos = positions[-1].item()
                                in_bounds = pos < hidden_states.shape[1]
                                print(f"    Position: {pos}, hidden_states.shape[1]: {hidden_states.shape[1]}, in_bounds: {in_bounds}")
                    self._debug_loop_summary = True
                
                # Debug: why no action_hidden_states?
                if len(action_hidden_states) == 0 and not hasattr(self, '_debug_action_empty'):
                    print(f"\n[Model Debug] No action_hidden_states collected!")
                    print(f"  batch_size: {batch_size}, seq_len: {seq_len}")
                    print(f"  hidden_states shape: {hidden_states.shape}")
                    print(f"  input_ids shape: {input_ids.shape}")
                    print(f"  action_token_positions shape: {action_token_positions.shape}")
                    print(f"  action_labels: {action_labels}")
                    self._debug_action_empty = True
                
                if len(action_hidden_states) > 0:
                    # Stack hidden states
                    action_hidden_states = torch.stack(action_hidden_states, dim=0)
                    
                    # Ensure dtype consistency and move to action_head device
                    action_hidden_states = action_hidden_states.to(
                        self.action_head.weight.dtype
                    ).to(self.action_head.weight.device)
                    
                    # Compute action logits: [num_valid_samples, 4]
                    action_logits = self.action_head(action_hidden_states)
                    
                    # Stack valid labels and move to same device as logits
                    valid_action_labels = torch.stack(valid_action_labels).to(action_logits.device)
                    
                    # Compute action loss
                    action_loss = self.action_loss_fn(action_logits, valid_action_labels)
        
        # Combine losses
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss
            if action_loss is not None:
                # Ensure action_loss is on the same device as lm_loss
                action_loss = action_loss.to(lm_loss.device)
                total_loss = total_loss + action_loss_weight * action_loss
        
        # Create base output
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (model_outputs.hidden_states if hasattr(model_outputs, 'hidden_states') else None,)
            if output_attentions:
                output = output + (model_outputs.attentions if hasattr(model_outputs, 'attentions') else None,)
            if total_loss is not None:
                output = (total_loss,) + output
            # Add action outputs as additional tuple elements
            if action_logits is not None:
                output = output + (action_logits,)
            if action_loss is not None:
                output = output + (action_loss,)
            return output
        
        # Create output with action information
        output = CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values if hasattr(model_outputs, 'past_key_values') else None,
            hidden_states=model_outputs.hidden_states if hasattr(model_outputs, 'hidden_states') else None,
            attentions=model_outputs.attentions if hasattr(model_outputs, 'attentions') else None,
        )
        
        # Add action outputs as attributes (for compatibility with training loops)
        output.action_logits = action_logits
        output.action_loss = action_loss
        
        return output
