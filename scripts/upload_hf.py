from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your dataset files
upload_folder(folder_path="/mnt/swx/ThinkVLN/results/watcher_rollout_train_full_out", repo_id="luna-shi/watcher_rollouts", repo_type="dataset")
