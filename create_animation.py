import imageio.v2 as imageio # Use imageio.v2 for consistency if used elsewhere
import pathlib
import os # For checking file size
import re # Import regular expressions

frame_dir = "frames"
output_path = "hotelling.gif"
fps = 2
frame_dir = pathlib.Path(frame_dir)

print(f"Looking for frames in: {frame_dir.resolve()}")

# Get all frame_*.png files
all_png_files = list(frame_dir.glob("frame_*.png"))

# Define a function to extract the numerical part of the filename
def extract_frame_number(file_path):
    match = re.search(r"frame_(\d+)\.png$", file_path.name)
    if match:
        return int(match.group(1))
    return -1 # Should not happen if files match frame_*.png

# Sort files based on the extracted frame number
png_files = sorted(all_png_files, key=extract_frame_number)


if not png_files:
    print(f"No PNG files found in {frame_dir}. Exiting.")
else:
    print(f"Found {len(png_files)} PNG files. Attempting to create GIF...")
    # Ensure the output directory exists if it's different
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output_file, mode='I', duration=int(1000/fps), loop=0) as writer: # duration in ms, loop=0 for infinite loop
        frames_processed_count = 0
        for png_path in png_files:
            try:
                # Check if the file is empty
                if os.path.getsize(png_path) == 0:
                    print(f"Skipping empty file: {png_path}")
                    continue

                print(f"Processing: {png_path}")
                image = imageio.imread(png_path)
                writer.append_data(image)
                frames_processed_count += 1
            except Exception as e:
                print(f"Could not read or process file {png_path}: {e}. Skipping.")

        if frames_processed_count > 0:
            print(f"Animation saved to {output_file} with {frames_processed_count} frames.")
        else:
            print(f"No valid frames were processed. Animation {output_file} might be empty or not created.")