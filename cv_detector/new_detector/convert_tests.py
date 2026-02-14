"""Convert test HEIC images to JPG."""
import pillow_heif
from PIL import Image
from pathlib import Path

pillow_heif.register_heif_opener()

test_dirs = ["test1", "test2", "test3"]
base = Path("/Users/adam/Desktop/TreeHacks26/cv_detector/new_detector")

for test_dir in test_dirs:
    dir_path = base / test_dir
    if not dir_path.exists():
        continue

    # Create output dir
    out_dir = dir_path / "jpg"
    out_dir.mkdir(exist_ok=True)

    # Convert each HEIC
    for heic_file in sorted(dir_path.glob("*.HEIC")):
        img = Image.open(heic_file)
        out_path = out_dir / f"{heic_file.stem}.jpg"
        img.save(out_path, "JPEG", quality=95)
        print(f"Converted: {heic_file.name} -> {out_path.name}")

print("\nDone!")
