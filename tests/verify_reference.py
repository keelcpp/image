#!/usr/bin/env python3
"""Differential reference verification for keel::image against MATLAB R2025b
Image Processing Toolbox and SciPy/NumPy.

Usage:
    python tests/verify_reference.py [path/to/test_image.exe]
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


def find_exe() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "build" / "Release" / "test_image.exe",
        root / "build" / "Debug" / "test_image.exe",
        root / "build" / "test_image.exe",
        root / "build" / "test_image",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not locate test_image binary; pass its path as argv[1]")


def parse_kv(text: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not v:
            continue
        parts = [p.strip() for p in v.split(",") if p.strip()]
        try:
            out[k] = np.array([float(p) for p in parts], dtype=np.float64)
        except ValueError:
            pass
    return out


MATLAB_SCRIPT = r"""
format long g;
pvec = @(name, arr) fprintf('%s=%s\n', name, strjoin(arrayfun(@(x) sprintf('%.15g', x), full(arr(:)).', 'UniformOutput', false), ','));
pcplx = @(name, arr) pvec(name, reshape([real(arr(:)).'; imag(arr(:)).'], 1, []));

% 1. Color conversions
r = [0.2 0.75 0.9]; g = [0.5 0.25 0.1]; b = [0.8 0.6 0.3];
rgb = cat(3, r, g, b);
pvec('rgb2gray', rgb2gray(rgb));
hsv = rgb2hsv(rgb);
pvec('rgb2hsv', [hsv(:,:,1) hsv(:,:,2) hsv(:,:,3)]);
hsv_rt = hsv2rgb(hsv);
pvec('hsv2rgb', [hsv_rt(:,:,1) hsv_rt(:,:,2) hsv_rt(:,:,3)]);
pvec('rgb2ntsc', rgb2ntsc([0.2 0.5 0.8]));
pvec('ntsc2rgb', ntsc2rgb([0.5 0.2 -0.1]));
ycb = rgb2ycbcr(rgb);
pvec('rgb2ycbcr', [ycb(:,:,1) ycb(:,:,2) ycb(:,:,3)]);
ycb_rt = ycbcr2rgb(ycb);
pvec('ycbcr2rgb', [ycb_rt(:,:,1) ycb_rt(:,:,2) ycb_rt(:,:,3)]);
xyz = rgb2xyz(rgb);
pvec('rgb2xyz', [xyz(:,:,1) xyz(:,:,2) xyz(:,:,3)]);
lab = rgb2lab(rgb);
pvec('rgb2lab', [lab(:,:,1) lab(:,:,2) lab(:,:,3)]);

% 2. Quality
A = reshape(sin(1:256)*0.5 + 0.5, 16, 16);
B = A * 0.9 + 0.05;
pvec('quality', [immse(A, B), psnr(A, B, 1.0), mean2(A), std2(A), ssim(A, B, 'DynamicRange', 1.0)]);

% 3. Otsu
[t_otsu, em_otsu] = otsuthresh([10 40 20 2 1 15 50 12]);
pvec('otsu', [t_otsu, em_otsu]);

% 4. Demosaic 6x6 RGGB (interior pixels avoid border convention differences)
M = uint8(reshape(1:36, 6, 6) * 5);
D = double(demosaic(M, 'rggb'));
R = D(:,:,1); G = D(:,:,2); Bch = D(:,:,3);
pvec('demosaic_interior', [R(15), R(22), G(15), G(22), Bch(15), Bch(22)]);

% 5. Texture
ti = reshape([0.0 0.25 0.5 0.75 0.25 0.5 0.75 1.0 0.5 0.75 1.0 0.0 0.75 1.0 0.0 0.25], 4, 4);
glcm = graycomatrix(ti, 'NumLevels', 4, 'GrayLimits', [0 1], 'Offset', [0 1]);
pvec('glcm', double(glcm));
gp = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});
pvec('glcm_props', [gp.Contrast, gp.Correlation, gp.Energy, gp.Homogeneity]);

% 6. Chromadapt
rgb_c = cat(3, [0.3 0.6 0.8], [0.4 0.5 0.2], [0.5 0.2 0.7]);
ca1 = chromadapt(rgb_c, [0.8 0.7 0.6], 'Method', 'bradford', 'ColorSpace', 'srgb');
pvec('chromadapt_bradford', [ca1(:,:,1) ca1(:,:,2) ca1(:,:,3)]);

% 7. Illumination + imcolordiff
rgb_i = cat(3, [0.2 0.4; 0.6 0.8], [0.1 0.3; 0.5 0.7], [0.3 0.5; 0.7 0.9]);
iw = illumwhite(rgb_i, 0);
ig1 = illumgray(rgb_i, 0, 'Norm', 1);
ig2 = illumgray(rgb_i, 0, 'Norm', 2);
pvec('illum', [iw, ig1, ig2]);
l1 = reshape([50 20 30], 1, 1, 3);
l2 = reshape([55 25 22], 1, 1, 3);
de94 = imcolordiff(l1, l2, 'Standard', 'CIE94', 'isInputLab', true);
de00 = imcolordiff(l1, l2, 'Standard', 'CIEDE2000', 'isInputLab', true);
de00c = imcolordiff(l1, l2, 'Standard', 'CIEDE2000', 'isInputLab', true, 'K1', 0.048, 'K2', 0.014);
pvec('colordiff', [de94, de00, de00c]);

% 8. Geometric
g4 = reshape(1:16, 4, 4);
pvec('resize_cubic', imresize(g4, [6 6], 'bicubic'));
pvec('rotate_30_bilin', imrotate(g4, 30, 'bilinear', 'loose'));
pvec('pyr_reduce', impyramid(g4, 'reduce'));

% 9. 2D FIR
hd_r = [0 0 0; 0 1 0; 0 0 0];
pvec('fsamp2_real', fsamp2(hd_r));
hd_c = [1+1i 2 3; 4 5-2i 6; 7 8 9];
pcplx('fsamp2_complex', fsamp2(hd_c));
pvec('ftrans2', ftrans2([0.1 0.2 0.4 0.2 0.1]));

% 10. Box / guided / contrast
a25 = reshape((1:25)/25, 5, 5);
pvec('imboxfilt', imboxfilt(a25, 3, 'Padding', 'replicate'));
pvec('imguidedfilter', imguidedfilter(a25, 'NeighborhoodSize', [3 3], 'DegreeOfSmoothing', 0.01));
pvec('stretchlim', stretchlim(a25, [0.01 0.99]));
pvec('imadjust', imadjust(a25, [0.2 0.8], [0.1 0.9], 1.5));
[mt2, mteff] = multithresh(a25, 2);
pvec('multithresh2', [mt2, mteff]);

% 11. Morphology
ma = reshape([0.1 0.4 0.2 0.3 0.5 0.9 0.6 0.2 0.3 0.7 0.8 0.4 0.2 0.3 0.5 0.1], 4, 4);
se = strel([0 1 0; 1 1 1; 0 1 0]);
pvec('imerode_f64', imerode(ma, se));
pvec('imdilate_f64', imdilate(ma, se));
pvec('imregionalmax', double(imregionalmax(ma, 8)));
pvec('imregionalmin', double(imregionalmin(ma, 8)));
pvec('imhmax', imhmax(ma, 0.25, 8));
pvec('imhmin', imhmin(ma, 0.25, 8));
hole = [0 0 0 0 0; 0 1 1 1 0; 0 1 0 1 0; 0 1 1 1 0; 0 0 0 0 0] > 0;
pvec('imfill_bin', double(imfill(hole, 4, 'holes')));

% 12. Filters
pvec('fspecial_gauss', fspecial('gaussian', [5 5], 1.2));
pvec('fspecial_log', fspecial('log', [5 5], 0.8));
pvec('fspecial_disk', fspecial('disk', 2.5));
pvec('fspecial_motion', fspecial('motion', 7.0, 35.0));
pvec('fspecial_unsharp', fspecial('unsharp', 0.3));

X = [1 2 3; 4 5 6; 7 8 9];
K = [1 2; 3 4];
pvec('imfilter_full_corr', imfilter(X, K, 0, 'full', 'corr'));
pvec('imfilter_full_conv', imfilter(X, K, 0, 'full', 'conv'));
pvec('imfilter_full_repl', imfilter(X, K, 'replicate', 'full', 'corr'));
pvec('imfilter_same_corr', imfilter(X, K, 0, 'same', 'corr'));
pvec('imfilter_same_conv', imfilter(X, K, 0, 'same', 'conv'));

a16 = reshape(sin((1:16)*0.7)*5 + 10, 4, 4);
pvec('medfilt2', medfilt2(a16, [3 3]));
pvec('stdfilt', stdfilt(a16, ones(3, 3)));
pvec('rangefilt', rangefilt(a16, ones(3, 3)));
[w2, wnoise] = wiener2(a16, [3 3]);
pvec('wiener2', w2);
pvec('wiener2_noise', wnoise);
pvec('convmtx2', convmtx2(K, 3, 3));

% 13. Transforms & deconvolution
pvec('dctmtx', dctmtx(4));
b16 = reshape((1:16)*0.5, 4, 4);
d2 = dct2(b16);
pvec('dct2', d2);
pvec('idct2', idct2(d2));
pvec('integral2d', integralImage(b16));
T2 = [1 2; 3 4];
B5 = ones(5, 5) * 0.3;
B5(2:3, 2:3) = [0.1 0.2; 0.3 0.4];
pvec('normxcorr2', normxcorr2(T2, B5));
psf = fspecial('gaussian', [3 3], 0.8);
pcplx('psf2otf', psf2otf(psf, [4 4]));
et_in = reshape(cos((0:63)*0.3)*0.5 + 0.5, 8, 8);
pvec('edgetaper', edgetaper(et_in, psf));
pvec('deconvwnr', deconvwnr(et_in, psf, 0.01));

% 14. Regions
bw = reshape([1 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1], 5, 5) > 0;
pvec('bwlabel8', bwlabel(bw, 8));
pvec('bwperim4', double(bwperim(bw, 4)));
pvec('bweuler', [bweuler(bw, 8), bweuler(bw, 4)]);
pvec('bwdist', double(bwdist(bw)));
rp = regionprops(bwlabel(bw, 8), 'Area','Centroid','MajorAxisLength','MinorAxisLength','Eccentricity','Orientation','EquivDiameter');
pvec('regionprops0', [rp(1).Area, rp(1).Centroid(1), rp(1).Centroid(2), rp(1).MajorAxisLength, rp(1).MinorAxisLength, rp(1).Eccentricity, rp(1).Orientation, rp(1).EquivDiameter]);

% 15. Features & segmentation
A12 = [1 4 9 16; 2 5 10 17; 3 6 11 18];
[gxs, gys] = imgradientxy(A12, 'sobel');
pvec('gradxy_sobel_x', gxs);
pvec('gradxy_sobel_y', gys);
[gxc, gyc] = imgradientxy(A12, 'central');
pvec('gradxy_central_x', gxc);
pvec('gradxy_central_y', gyc);

sq = zeros(8, 8); sq(3:6, 3:6) = 1;
pvec('corner_harris', cornermetric(sq, 'Harris', 'SensitivityFactor', 0.04));
pvec('poly2mask', double(poly2mask([2 5 5 2], [2 2 5 5], 6, 6)));
gimg = reshape(1 + 0.2*(0:15), 4, 4);
mask0 = false(4, 4); mask0(1, 1) = true;
pvec('graydist_qe', graydist(gimg, mask0, 'quasi-euclidean'));

hf = zeros(5, 5);
for c = 0:4
    for r = 0:4
        hf(r+1, c+1) = r + 2*c;
    end
end
hm = false(5, 5); hm(3, 3) = true; hm(4, 3) = true;
pvec('regionfill', regionfill(hf, hm));

gw_in = zeros(8, 8);
for c = 0:7
    for r = 0:7
        gw_in(r+1, c+1) = sin(r*0.5) + cos(c*0.3);
    end
end
pvec('gradientweight', gradientweight(gw_in, 1.5, 'RolloffFactor', 2.0, 'WeightCutoff', 0.1));

fprintf('MATLAB_DONE=1\n');
"""


def run_matlab() -> dict[str, np.ndarray]:
    proc = subprocess.run(
        ["matlab", "-batch", MATLAB_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    kv = parse_kv(proc.stdout)
    if "MATLAB_DONE" not in kv or kv["MATLAB_DONE"][0] != 1.0:
        raise RuntimeError(
            f"MATLAB reference script did not reach MATLAB_DONE=1 (rc={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return kv


def main() -> int:
    exe = find_exe()
    proc = subprocess.run([str(exe), "--dump-ref"], capture_output=True, text=True, check=True)
    keel_kv = parse_kv(proc.stdout)
    matlab_kv = run_matlab()

    # Custom tolerances per test key (abs_tol, rel_tol)
    tols: dict[str, tuple[float, float]] = {
        "rgb2gray": (1e-12, 1e-12),
        "rgb2hsv": (1e-12, 1e-12),
        "hsv2rgb": (1e-12, 1e-12),
        "rgb2ntsc": (1e-12, 1e-12),
        "ntsc2rgb": (1e-12, 1e-12),
        "rgb2ycbcr": (1e-12, 1e-12),
        "ycbcr2rgb": (1e-12, 1e-12),
        "rgb2xyz": (1e-5, 1e-5),     # sRGB matrix rounding precision
        "rgb2lab": (2e-3, 1e-4),     # D65 whitepoint / sRGB matrix digits
        "quality": (1e-6, 1e-6),
        "otsu": (1e-12, 1e-12),
        "demosaic_interior": (1.0, 0.0),  # MATLAB rounds uint8 vs keel f64
        "glcm": (1e-12, 1e-12),
        "glcm_props": (1e-12, 1e-12),
        "chromadapt_bradford": (5e-3, 5e-3),
        "illum": (2e-5, 2e-5),       # MATLAB illumwhite uses 16-bit histogram quantization (1/65535)
        "colordiff": (1e-7, 1e-7),   # MATLAB imcolordiff casts custom K1/K2 through single precision
        "resize_cubic": (1e-11, 1e-11),
        "rotate_30_bilin": (1e-11, 1e-11),
        "pyr_reduce": (1e-11, 1e-11),
        "fsamp2_real": (1e-12, 1e-12),
        "fsamp2_complex": (1e-12, 1e-12),
        "ftrans2": (1e-12, 1e-12),
        "imboxfilt": (1e-12, 1e-12),
        "imguidedfilter": (1e-10, 1e-10),
        "stretchlim": (1e-4, 1e-4),
        "imadjust": (1e-12, 1e-12),
        "multithresh2": (1e-11, 1e-11),
        "imerode_f64": (1e-12, 1e-12),
        "imdilate_f64": (1e-12, 1e-12),
        "imregionalmax": (0.0, 0.0),
        "imregionalmin": (0.0, 0.0),
        "imhmax": (1e-12, 1e-12),
        "imhmin": (1e-12, 1e-12),
        "imfill_bin": (0.0, 0.0),
        "fspecial_gauss": (1e-12, 1e-12),
        "fspecial_log": (1e-12, 1e-12),
        "fspecial_disk": (1e-12, 1e-12),
        "fspecial_motion": (1e-12, 1e-12),
        "fspecial_unsharp": (1e-12, 1e-12),
        "imfilter_full_corr": (1e-12, 1e-12),
        "imfilter_full_conv": (1e-12, 1e-12),
        "imfilter_full_repl": (1e-12, 1e-12),
        "imfilter_same_corr": (1e-12, 1e-12),
        "imfilter_same_conv": (1e-12, 1e-12),
        "medfilt2": (1e-12, 1e-12),
        "stdfilt": (1e-11, 1e-11),
        "rangefilt": (1e-12, 1e-12),
        "wiener2": (1e-11, 1e-11),
        "wiener2_noise": (1e-11, 1e-11),
        "convmtx2": (1e-12, 1e-12),
        "dctmtx": (1e-12, 1e-12),
        "dct2": (1e-11, 1e-11),
        "idct2": (1e-11, 1e-11),
        "integral2d": (1e-12, 1e-12),
        "normxcorr2": (5e-8, 1e-10), # keel zeroes constant patches exactly; MATLAB leaks ~1.9e-8 noise
        "psf2otf": (1e-12, 1e-12),
        "edgetaper": (1e-11, 1e-11),
        "deconvwnr": (1e-10, 1e-10),
        "bwlabel8": (0.0, 0.0),
        "bwperim4": (0.0, 0.0),
        "bweuler": (0.0, 0.0),
        "bwdist": (1e-12, 1e-12),
        "regionprops0": (1e-10, 1e-10),
        "gradxy_sobel_x": (1e-12, 1e-12),
        "gradxy_sobel_y": (1e-12, 1e-12),
        "gradxy_central_x": (1e-12, 1e-12),
        "gradxy_central_y": (1e-12, 1e-12),
        "corner_harris": (1e-8, 1e-8),
        "poly2mask": (0.0, 0.0),
        "graydist_qe": (1e-6, 1e-6),  # Dijkstra vs MATLAB chamfer quasi-euclidean
        "regionfill": (1e-8, 1e-8),
        "gradientweight": (1e-10, 1e-10),
    }

    failures = 0
    total = 0
    for key, (atol, rtol) in tols.items():
        total += 1
        if key not in keel_kv:
            print(f"FAIL [{key}]: missing from keel output")
            failures += 1
            continue
        if key not in matlab_kv:
            print(f"FAIL [{key}]: missing from MATLAB output")
            failures += 1
            continue
        a = keel_kv[key]
        b = matlab_kv[key]
        if a.shape != b.shape:
            print(f"FAIL [{key}]: shape mismatch keel={a.shape} vs matlab={b.shape}")
            failures += 1
            continue
        diff = np.abs(a - b)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        ok = np.all(diff <= (atol + rtol * np.abs(b)))
        if ok:
            print(f"PASS [{key:22s}] n={a.size:3d} max_abs={max_abs:.3e}")
        else:
            idx = int(np.argmax(diff))
            print(
                f"FAIL [{key:22s}] max_abs={max_abs:.3e} at idx={idx}: "
                f"keel={a[idx]:.15g} vs matlab={b[idx]:.15g}"
            )
            failures += 1

    # Independent SciPy ndimage cross-check for Euclidean distance transform
    bw = np.array(
        [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
        dtype=np.uint8,
    ).reshape((5, 5), order="F")
    scipy_edt = ndimage.distance_transform_edt(bw == 0).ravel(order="F")
    total += 1
    edt_err = float(np.max(np.abs(keel_kv["bwdist"] - scipy_edt)))
    if edt_err <= 1e-12:
        print(f"PASS [{'scipy_bwdist':22s}] n={scipy_edt.size:3d} max_abs={edt_err:.3e}")
    else:
        print(f"FAIL [{'scipy_bwdist':22s}] max_abs={edt_err:.3e}")
        failures += 1

    print(f"\nReference verification: {total - failures}/{total} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
