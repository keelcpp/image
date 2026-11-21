# keel::image

Zero-STL C++20 image processing kernels built on [`keel`](https://github.com/keelcpp/keel).

> Note: Image file codecs (PNG, JPEG, TIFF, BMP, TGA, PNM, GIF) live in [`keelcpp/imagefile`](https://github.com/keelcpp/imagefile).

## Features

- **Color-space conversions**: RGB↔Gray (BT.601 luma), RGB↔HSV, RGB↔NTSC (YIQ), RGB↔YCbCr (BT.601 studio swing), sRGB gamma, RGB↔XYZ (D65), XYZ↔CIELAB (D65), RGB↔CIELAB — both per-pixel and planar batch APIs.
- **Image quality & 2D filtering**: `mse`, `mse_c64`, `psnr`, `mean2`, `std2`, `gaussian_kernel_2d`, `correlate_2d_replicate`, and `ssim`.
- **Histograms & thresholding**: `histogram` and `otsu_threshold` (Otsu's method with effectiveness metric).

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
