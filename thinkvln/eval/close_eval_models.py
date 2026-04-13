import json
import os
import sys
from typing import Optional

import torch

from thinkvln.engine.inference import load_model_and_processor
from thinkvln.models.navigation_model import (
    NavigationModel,
    StreamVLNNavigationModel,
    ThinkVLNActorNavigationModel,
    ThinkVLNFMNavigationModel,
    ThinkVLNNavigationModel,
)


def _load_streamvln_pretrained_model(
    streamvln_cls,
    model_path: str,
    config,
    dtype,
    prefer_flash_attention: bool = True,
):
    kwargs = dict(
        dtype=dtype,
        config=config,
        low_cpu_mem_usage=False,
    )
    attn_impl = "flash_attention_2" if prefer_flash_attention else "eager"
    try:
        return streamvln_cls.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
            **kwargs,
        )
    except ValueError as exc:
        message = str(exc)
        if (not prefer_flash_attention) or ("Flash Attention 2" not in message and "flash_attention_2" not in message):
            raise
        return streamvln_cls.from_pretrained(
            model_path,
            attn_implementation="eager",
            **kwargs,
        )


def load_thinkvln_actor_model(
    model_path: str,
    device: str = "cuda",
    base_model_path: Optional[str] = None,
):
    from transformers import AutoProcessor
    from thinkvln.models.actor_config import ThinkVLNActorConfig
    from thinkvln.models.thinkvln_actor import ThinkVLNActor

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)

    if is_lora:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        actual_base_model = base_model_path or adapter_config.get("base_model_name_or_path")
        if not actual_base_model:
            raise ValueError(
                "Base model path is required for LoRA actor checkpoint. "
                "Use --base_model_path or provide base_model_name_or_path in adapter_config.json."
            )

        processor = AutoProcessor.from_pretrained(actual_base_model, trust_remote_code=True)

        actor_config_file = os.path.join(model_path, "actor_config.json")
        if os.path.isfile(actor_config_file):
            with open(actor_config_file, "r", encoding="utf-8") as f:
                actor_cfg = ThinkVLNActorConfig(**json.load(f))
        else:
            actor_cfg = ThinkVLNActorConfig()

        model = ThinkVLNActor.from_pretrained(
            actual_base_model,
            actor_config=actor_cfg,
            device_map="cpu",
            dtype=torch.bfloat16,
        )

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path)
        model = model.to(dtype=torch.bfloat16).to(device)
    else:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = ThinkVLNActor.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
        ).to(device)

    model.requires_grad_(False)
    model.eval()
    return model, processor


def build_nav_model(args, device: str, rank: int, world_size: int) -> NavigationModel:
    print(f"Loading {args.model_type} model from {args.model_path}...")

    if args.model_type == "thinkvln":
        model, processor = load_model_and_processor(args.model_path, device)
        model.requires_grad_(False)
        model.eval()
        return ThinkVLNNavigationModel(
            model=model,
            processor=processor,
            device=str(device),
            max_new_tokens=args.model_max_length,
        )

    if args.model_type == "thinkvln_actor":
        model, processor = load_thinkvln_actor_model(
            model_path=args.model_path,
            device=str(device),
            base_model_path=args.base_model_path,
        )
        return ThinkVLNActorNavigationModel(
            model=model,
            processor=processor,
            device=str(device),
            memory_num_history_images=getattr(args, "memory_num_history_images", 6),
            done_threshold=getattr(args, "done_threshold", 0.85),
            use_memory=getattr(args, "use_memory", True),
        )

    if args.model_type == "thinkvln_fm_actor":
        from transformers import AutoProcessor
        from thinkvln.models.thinkvln_fm_actor import ThinkVLNFMActor

        adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)

        if is_lora:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            actual_base_model = args.base_model_path or adapter_config.get("base_model_name_or_path")
            if not actual_base_model:
                raise ValueError(
                    "Base model path is required for LoRA FM checkpoint. "
                    "Use --base_model_path or include base_model_name_or_path in adapter_config.json."
                )
            processor = AutoProcessor.from_pretrained(actual_base_model, trust_remote_code=True)
            model = ThinkVLNFMActor.from_pretrained(
                actual_base_model,
                device_map="cpu",
                dtype=torch.bfloat16,
            )
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.model_path)
            model = model.to(dtype=torch.bfloat16).to(device)
        else:
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
            model = ThinkVLNFMActor.from_pretrained(
                args.model_path,
                dtype=torch.bfloat16,
            ).to(device)

        model.requires_grad_(False)
        model.eval()
        return ThinkVLNFMNavigationModel(
            model=model,
            processor=processor,
            device=str(device),
            memory_num_history_images=getattr(args, "memory_num_history_images", 6),
            done_threshold=getattr(args, "done_threshold", 0.85),
            use_memory=getattr(args, "use_memory", True),
        )

    if args.model_type in {"streamvln", "streamvln_actor"}:
        thinkvln_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        possible_llava_paths = [
            os.path.join(thinkvln_root, "third_party", "LLaVA-NeXT"),
            os.path.join(thinkvln_root, "LLaVA-NeXT"),
            os.path.expanduser("~/LLaVA-NeXT"),
        ]
        if thinkvln_root not in sys.path:
            sys.path.insert(0, thinkvln_root)

        llava_found = False
        for llava_path in possible_llava_paths:
            if os.path.exists(llava_path) and os.path.exists(os.path.join(llava_path, "llava")):
                if llava_path not in sys.path:
                    sys.path.insert(0, llava_path)
                compat_patch_path = os.path.join(llava_path, "llava", "compat_patch.py")
                if os.path.exists(compat_patch_path):
                    import importlib.util

                    spec = importlib.util.spec_from_file_location("llava.compat_patch", compat_patch_path)
                    compat_patch = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(compat_patch)
                llava_found = True
                break
        if not llava_found:
            raise ImportError("LLaVA-NeXT not found. StreamVLN requires LLaVA-NeXT codebase.")

        import transformers
        from streamvln.model.stream_video_vln import StreamVLNForCausalLM
        adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)

        if is_lora:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            actual_base_model = args.base_model_path or adapter_config.get("base_model_name_or_path")
            if not actual_base_model:
                raise ValueError(
                    "Base model path is required for LoRA StreamVLN checkpoint. "
                    "Use --base_model_path or provide base_model_name_or_path in adapter_config.json."
                )
            # Keep tokenizer aligned with the LoRA checkpoint artifacts.
            tokenizer_source = args.model_path
            config_source = actual_base_model
        else:
            actual_base_model = args.model_path
            tokenizer_source = args.model_path
            config_source = args.model_path

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_source,
            model_max_length=args.model_max_length,
            padding_side="right",
        )
        config = transformers.AutoConfig.from_pretrained(config_source)

        if not hasattr(config, "layer_types") or config.layer_types is None:
            num_layers = getattr(config, "num_hidden_layers", 32)
            sliding_window = getattr(config, "sliding_window", None)
            max_window_layers = getattr(config, "max_window_layers", num_layers)
            if sliding_window is not None:
                config.layer_types = [
                    "sliding_attention" if i >= max_window_layers else "full_attention"
                    for i in range(num_layers)
                ]
            else:
                config.layer_types = ["full_attention"] * num_layers

        model = _load_streamvln_pretrained_model(
            streamvln_cls=StreamVLNForCausalLM,
            model_path=actual_base_model,
            config=config,
            dtype=torch.bfloat16,
            prefer_flash_attention=bool(torch.cuda.is_available()),
        )
        if is_lora:
            model.resize_token_embeddings(len(tokenizer))
        if is_lora:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.model_path)
            non_lora_path = os.path.join(args.model_path, "non_lora_trainables.bin")
            if os.path.exists(non_lora_path):
                non_lora_state = torch.load(non_lora_path, map_location="cpu")
                model.load_state_dict(non_lora_state, strict=False)
        core_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        core_model.model.num_history = args.num_history
        model.requires_grad_(False)
        model.to(device)
        model.eval()
        model.reset(world_size)

        return StreamVLNNavigationModel(
            model=model,
            tokenizer=tokenizer,
            device=str(device),
            num_frames=args.num_frames,
            num_future_steps=args.num_future_steps,
            num_history=args.num_history,
            env_id=rank,
            done_threshold=getattr(args, "done_threshold", 0.85),
            include_previous_progress_in_prompt=False,
        )

    raise ValueError(f"Unknown model type: {args.model_type}")


__all__ = [
    "load_thinkvln_actor_model",
    "build_nav_model",
    "_load_streamvln_pretrained_model",
]
