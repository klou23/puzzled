"""Convert HEIC images to JPG format."""

import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def convert_heic_to_jpg(input_dir: str, output_dir: str):
    """Convert all HEIC files in input_dir to JPG in output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.heic'):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.rsplit('.', 1)[0] + '.jpg'
            output_path = os.path.join(output_dir, output_filename)

            try:
                img = Image.open(input_path)
                img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=95)
                print(f"Converted: {filename} -> {output_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "new_steps")
    output_dir = os.path.join(script_dir, "steps_jpg")
    convert_heic_to_jpg(input_dir, output_dir)
    print(f"\nConversion complete! JPGs saved to: {output_dir}")
