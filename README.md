# keel::image

Zero-STL C++20 image processing kernels built on [`keel`](https://github.com/keelcpp/keel).

> Note: Image file codecs (PNG, JPEG, TIFF, BMP, TGA, PNM, GIF) live in [`keelcpp/imagefile`](https://github.com/keelcpp/imagefile).

## Features

- **Color-space conversions & white balance**: RGB↔Gray (BT.601 luma), RGB↔HSV, RGB↔NTSC (YIQ), RGB↔YCbCr (BT.601 studio swing), sRGB gamma, RGB↔XYZ (D65), XYZ↔CIELAB (D65), RGB↔CIELAB, Malvar-He-Cutler Bayer demosaicing (`demosaic_mhc`), and chromatic adaptation (`chromadapt_planar`: Bradford / Von Kries / Simple across sRGB, Adobe RGB, ProPhoto RGB, Linear RGB).
- **Image quality, texture & filtering**: `mse`, `mse_c64`, `psnr`, `mean2`, `std2`, `gaussian_kernel_2d`, `correlate_2d_replicate`, `ssim`, GLCM & Haralick properties (`graycomatrix`, `graycoprops`), 3D filter kernels & 2D windowed FIR (`fspecial3_*`, `fwind2`), multiscale 2D/3D Frangi vesselness (`fibermetric_2d`, `fibermetric_3d`), and Fast Local Laplacian Filtering (`local_laplacian_gray`, `local_laplacian_rgb`).
- **Histograms & thresholding**: `histogram` and `otsu_threshold` (Otsu's method with effectiveness metric).

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
