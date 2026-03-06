import numpy as np

from bat_tracker.valid_region import (
    build_mask,
    detect_valid_region,
    estimate_illumination,
    generate_valid_region_mask,
    horizontal_profile,
)


def test_detect_valid_region_synthetic_vignette_recovers_central_band():
    h, w = 120, 300
    x = np.arange(w, dtype=np.float32)
    center = (w - 1) / 2.0
    # Bright center with darker sides to emulate lateral IR falloff.
    profile = 30.0 + 200.0 * np.exp(-((x - center) ** 2) / (2.0 * (0.22 * w) ** 2))
    image = np.repeat(profile[np.newaxis, :], h, axis=0).astype(np.uint8)

    illum = estimate_illumination(image, blur_kernel_size=31)
    _, smooth = horizontal_profile(illum, profile_smooth_window=21)
    x_start, x_end = detect_valid_region(
        smoothed_profile=smooth,
        threshold_ratio=0.45,
        safety_margin=5,
        min_region_width_ratio=0.35,
    )

    assert x_start < x_end
    center_x = w // 2
    assert x_start < center_x < x_end
    width = x_end - x_start + 1
    assert width >= int(0.35 * w)
    assert x_start > 0
    assert x_end < (w - 1)


def test_detect_valid_region_fallback_on_flat_low_contrast_image():
    h, w = 100, 280
    image = np.full((h, w), 40, dtype=np.uint8)
    mask, x_start, x_end = generate_valid_region_mask(
        image=image,
        config={
            "blur_kernel_size": 31,
            "profile_smooth_window": 11,
            "threshold_ratio": 0.45,
            "safety_margin": 10,
            "min_region_width_ratio": 0.35,
        },
    )

    assert mask.shape == image.shape
    assert mask.dtype == np.uint8
    assert x_start < x_end
    width = x_end - x_start + 1
    assert int(0.6 * w) <= width <= int(0.8 * w)

    rebuilt = build_mask(image.shape, x_start, x_end)
    assert np.array_equal(mask, rebuilt)
