---
pipeline_tag: robotics
library_name: transformers
license: cc-by-nc-sa-4.0
tags:
  - vision-language-model
  - video-language-model
  - navigation
---

# StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling

<div id="top" align="center">
[![arXiv](https://img.shields.io/badge/arXiv-red?logo=arxiv)](http://arxiv.org/abs/2507.05240)
[![Project Page](https://img.shields.io/badge/Project-0065D3?logo=rocket&logoColor=white)](https://streamvln.github.io/)
[![Video Demo](https://img.shields.io/badge/Video-D33846?logo=youtube)](https://www.youtube.com/watch?v=gG3mpefOBjc)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/OpenRobotLab/StreamVLN)
</div>

<div style="text-align: center;">
    <img src="https://github.com/OpenRobotLab/StreamVLN/raw/main/assets/teaser.gif" width=100% >
</div>

## Paper
[StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling](https://huggingface.co/papers/2507.05240)

## Abstract
Vision-and-Language Navigation (VLN) in real-world settings requires agents to process continuous visual streams and generate actions with low latency grounded in language instructions. While Video-based Large Language Models (Video-LLMs) have driven recent progress, current VLN methods based on Video-LLM often face trade-offs among fine-grained visual understanding, long-term context modeling and computational efficiency. We introduce StreamVLN, a streaming VLN framework that employs a hybrid slow-fast context modeling strategy to support multi-modal reasoning over interleaved vision, language and action inputs. The fast-streaming dialogue context facilitates responsive action generation through a sliding-window of active dialogues, while the slow-updating memory context compresses historical visual states using a 3D-aware token pruning strategy. With this slow-fast design, StreamVLN achieves coherent multi-turn dialogue through efficient KV cache reuse, supporting long video streams with bounded context size and inference cost. Experiments on VLN-CE benchmarks demonstrate state-of-the-art performance with stable low latency, ensuring robustness and efficiency in real-world deployment.

## About
**StreamVLN** generates action outputs from continuous video input in an online, multi-turn dialogue manner. Built on **LLaVA-Video** as the foundational Video-LLM, we extend it for interleaved vision, language, and action modeling. For both effective context modeling of long sequence and efficient computation for real-time interaction, StreamVLN has: (1) a **fast-streaming** dialogue context with a sliding-window KV cache; and (2) a **slow-updating** memory via token pruning.

## Model Zoo

We provide two model checkpoints for different use cases:

-   **Benchmark Reproduction**
    Use this [checkpoint](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln) to reproduce results on the VLN-CE benchmark.

-   **Real-World Deployment**
    This [checkpoint](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world) is recommended for deployment on physical robots.

    We made two modifications:
    1.  **Remove redundant initial turn actions**: The initial left/right turns not mentioned in the instructions are removed for better instruction alignment.
    2.  **Trajectory safety**: Enhanced obstacle avoidance ensures more reliable navigation in real-world environments.

## Usage (with Transformers)

You can load StreamVLN models using the `transformers` library. Ensure you have the necessary dependencies installed as outlined in the [project's GitHub repository](https://github.com/OpenRobotLab/StreamVLN).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# Load model and processor
model_id = "mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # Adjust dtype based on your hardware (e.g., torch.float16 for Ampere GPUs)
    device_map="auto",
    trust_remote_code=True # Required for custom modeling components like Qwen-VL
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Example: This model is designed for Vision-and-Language Navigation (VLN).
# The full inference loop involves continuous visual stream processing and action generation
# within an environment. The snippet below shows a basic setup for text-image input.
# For complete VLN usage, including environment setup and action generation,
# please refer to the project's [GitHub repository](https://github.com/OpenRobotLab/StreamVLN).

# Load a sample image (replace with actual environment image in VLN tasks)
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?width=400"
image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

# Prepare text input using the chat template
messages = [
    {"role": "user", "content": "What is in the image? Describe it."},
]
text_input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Process inputs (text and image)
inputs = processor(text=text_input, images=image, return_tensors="pt").to(model.device)

# Generate response
output_ids = model.generate(
    **inputs,
    max_new_tokens=256, # Increase max_new_tokens for more detailed responses
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
)

# Decode and print the output, skipping the input prompt
output_text = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(output_text)
```

## Citation

If you find our work helpful, please consider starring this repo 🌟 and cite:

```bibtex
@misc{wei2025streamvlnstreamingvisionandlanguagenavigation,
      title={StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling}, 
      author={Meng Wei and Chenyang Wan and Xiqian Yu and Tai Wang and Yuqiang Yang and Xiaohan Mao and Chenming Zhu and Wenzhe Cai and Hanqing Wang and Yilun Chen and Xihui Liu and Jiangmiao Pang},
      year={2025},
      eprint={2507.05240},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.05240}, 
}
```

## License
This work is under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgements
This repository is based on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).