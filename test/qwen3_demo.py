from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Load model and processor from local path
model_path = "/mnt/swx/ThinkVLN/model_weights/qwen3vl-2"
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# Prepare messages with image and text
messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

# Apply chat template and prepare inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

print(f"inputs shape: {inputs.pixel_values.shape}")
print(f"inputs image_grid_thw shape: {inputs.image_grid_thw.shape}")

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)
