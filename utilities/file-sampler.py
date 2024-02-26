import os
import random
import shutil

SOURCE = 'raw-datasets/vgmusic-snes-pianified'
DESTINATION = 'sampled-datasets/snes'
SAMPLE_SIZE = 500


def main():
    if not os.path.exists(DESTINATION):
        os.mkdir(DESTINATION)

    file_count = 0
    sorted_files = []
    for root, dirs, files in os.walk(SOURCE):
        for file in files:
            if file.endswith('.mid') and not file.startswith('Aug-'):
                file_path = os.path.join(root, file)
                sorted_files.append(file_path)
                file_count += 1

    sorted_files = sorted(sorted_files)
    random_file_sample = random.sample(sorted_files, SAMPLE_SIZE)
    num_copied = 0
    for src_path in random_file_sample:
        path, file_name = os.path.split(src_path)
        dest_path = os.path.join(DESTINATION, file_name)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
            num_copied += 1

    print(f"Detected {file_count} files.")
    print(f"Copied {num_copied} random files to {DESTINATION}.")


if __name__ == "__main__":
    main()
