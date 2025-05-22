import imageio.v2 as imageio # Use imageio.v2 for consistency if used elsewhere
import pathlib
import os # For checking file size

frame_dir = "frames"
output_path = "hotelling.gif"
fps = 2
frame_dir = pathlib.Path(frame_dir)

print(f"Looking for frames in: {frame_dir.resolve()}")
png_files = sorted(frame_dir.glob("frame_*.png"))

if not png_files:
    print(f"No PNG files found in {frame_dir}. Exiting.")
else:
    print(f"Found {len(png_files)} PNG files. Attempting to create GIF...")
    with imageio.get_writer(output_path, mode='I', duration=1000/fps, loop=0) as writer: # duration in ms, loop=0 for infinite loop
        for png_path in png_files:
            try:
                # Check if the file is empty
                if os.path.getsize(png_path) == 0:
                    print(f"Skipping empty file: {png_path}")
                    continue

                print(f"Processing: {png_path}")
                image = imageio.imread(png_path)
                writer.append_data(image)
            except Exception as e:
                print(f"Could not read or process file {png_path}: {e}. Skipping.")
    print(f"Animation saved to {output_path}")