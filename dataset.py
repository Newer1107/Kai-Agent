from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="YashJain/UI-Elements-Detection-Dataset",
    repo_type="dataset",
    allow_patterns="yolo_dataset/*",
    local_dir="../yolo_dataset",   # ← parent folder
    local_dir_use_symlinks=False   # ← important on Windows
)

print("Downloaded YOLO dataset to:", local_path)