import random

import numpy as np
from PIL import Image

from smearlens.augment import augment, lab_stats, reinhard, stain_jitter


def smear():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(80, 230, (64, 64, 3), dtype=np.uint8))


def test_stain_jitter_identity_when_no_jitter():
    image = smear()
    out = stain_jitter(image, random.Random(0), sigma=0.0, bias=0.0)
    assert np.abs(np.asarray(out, int) - np.asarray(image, int)).max() <= 2


def test_augment_keeps_size_and_changes_pixels():
    image = smear()
    out = augment(image, random.Random(1))
    assert out.size == image.size
    assert not np.array_equal(np.asarray(out), np.asarray(image))


def test_reinhard_matches_target_stats():
    source = smear()
    mean, std = lab_stats([smear().point(lambda v: v // 2 + 60)])
    out = reinhard(source, mean, std)
    got_mean, _ = lab_stats([out])
    assert np.abs(got_mean - mean).max() < 3
