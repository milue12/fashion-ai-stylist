# 파일명: utils.py
from PIL import Image, ImageOps

def load_image(path, target_size=336):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = ImageOps.contain(img, (target_size, target_size))
    return img

def crop_item(image_path_or_pil, box):
    if isinstance(image_path_or_pil, str):
        full_image = Image.open(image_path_or_pil).convert("RGB")
    else:
        full_image = image_path_or_pil
    x, y, w, h = box
    return full_image.crop((x, y, x + w, y + h))