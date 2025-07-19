import os
import random

def split_dataset(image_dir, test_ratio=0.1, val_ratio=0.1, seed=42):
    # Load and shuffle filenames
    all_files = sorted(os.listdir(image_dir))
    random.seed(seed)
    random.shuffle(all_files)

    # Split
    total = len(all_files)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size

    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]

    print(f"Total: {total}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Save splits to text files
    with open("train_files.txt", "w") as f:
        f.write("\n".join(train_files))
    with open("val_files.txt", "w") as f:
        f.write("\n".join(val_files))
    with open("test_files.txt", "w") as f:
        f.write("\n".join(test_files))

    return train_files, val_files, test_files


if __name__ == "__main__":
    image_dir = "clean_reconstruction"
    split_dataset(image_dir, test_ratio=0.1, val_ratio=0.1)
