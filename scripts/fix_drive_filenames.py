import os
import re

# Resolve the preprocessed directory path relative to the repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
base_path = os.path.join(REPO_ROOT, "data", "preprocessed")

print(f"Target preprocessed path: {base_path}")

def sanitize_filename(filename: str) -> str:
    if "__" in filename:
        old_pid, img_stem = filename.split("__", 1)
        # Remove leading/trailing spaces, hyphens, and underscores
        new_pid = old_pid.strip(" -_")
        new_pid = re.sub(r'[^a-zA-Z0-9]+', '_', new_pid)
        if not new_pid or not new_pid[0].isalnum():
            new_pid = "patient_" + new_pid
        return f"{new_pid}__{img_stem}"
    return filename

splits = ["train", "val", "test"]
classes = ["Negatives", "Other", "Positives"]

total_renamed = 0

for split in splits:
    for cls in classes:
        dir_path = os.path.join(base_path, split, cls)
        if not os.path.exists(dir_path):
            continue
        print(f"Checking directory: {split}/{cls}...")
        count = 0
        for filename in os.listdir(dir_path):
            new_filename = sanitize_filename(filename)
            if filename != new_filename:
                old_file_path = os.path.join(dir_path, filename)
                new_file_path = os.path.join(dir_path, new_filename)
                
                # Check if the renamed file already exists to avoid overwrite conflicts
                if os.path.exists(new_file_path):
                    # If target already exists, just remove the old duplicate
                    os.remove(old_file_path)
                else:
                    os.rename(old_file_path, new_file_path)
                count += 1
        print(f"  -> Renamed/cleaned {count} files in {split}/{cls}.")
        total_renamed += count

print("\n==================================================")
print(f"SUCCESS: Total {total_renamed} files sanitized on disk!")
print("==================================================")
