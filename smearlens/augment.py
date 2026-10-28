import random

import numpy as np
from PIL import Image, ImageEnhance

# Ruifrok & Johnston stain vectors (haematoxylin, eosin, DAB), as in scikit-image.
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def stain_jitter(image: Image.Image, rng: random.Random, sigma: float = 0.05, bias: float = 0.05) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    od = -np.log(np.clip(rgb, 1e-6, 1.0))
    hed = od.reshape(-1, 3) @ HED_FROM_RGB
    alpha = np.array([rng.uniform(1 - sigma, 1 + sigma) for _ in range(3)])
    beta = np.array([rng.uniform(-bias, bias) for _ in range(3)])
    od = (hed * alpha + beta) @ RGB_FROM_HED
    out = np.exp(-od).reshape(rgb.shape)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def augment(image: Image.Image, rng: random.Random, strength: float = 1.0) -> Image.Image:
    image = stain_jitter(image, rng, sigma=0.05 * strength, bias=0.05 * strength)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        image = enhancer(image).enhance(rng.uniform(1 - 0.2 * strength, 1 + 0.2 * strength))
    scale = rng.uniform(1 - 0.15 * strength, 1 + 0.15 * strength)
    w, h = image.size
    resized = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    background = tuple(int(v) for v in np.median(np.asarray(image).reshape(-1, 3), axis=0))
    canvas = Image.new("RGB", (w, h), background)
    canvas.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2))
    if rng.random() < 0.5:
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
    return canvas.rotate(rng.choice([0, 90, 180, 270]))


def reinhard(image: Image.Image, target_mean: np.ndarray, target_std: np.ndarray) -> Image.Image:
    lab = np.asarray(image.convert("LAB"), dtype=np.float64)
    mean, std = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    out = (lab - mean) / std * target_std + target_mean
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="LAB").convert("RGB")


def lab_stats(images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.concatenate([np.asarray(i.convert("LAB"), dtype=np.float64).reshape(-1, 3) for i in images])
    return pixels.mean(0), pixels.std(0)
