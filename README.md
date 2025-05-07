# Fourier Transform Watermarking

This project implements a digital watermarking system using the Discrete Fourier Transform (DFT) for grayscale images.

## Overview

Digital watermarking embeds hidden data into media files to protect copyright or verify authenticity. This implementation uses frequency domain watermarking by embedding the watermark in the Fourier spectrum of the original image.

## Key Features

- Embed watermarks into grayscale images using DFT
- Extract watermarks from watermarked images
- Visualize frequency spectrum before and after watermarking
- Compare original and watermarked images

## Requirements

- Python 3.x
- NumPy
- OpenCV (cv2)
- Matplotlib

You can install the required packages using:

```
pip install numpy opencv-python matplotlib
```

## Usage

1. Create an `input` directory in the project root
2. Place your original image as `input/original.png`
3. Place your watermark image as `input/watermark.png`
4. Edit `main.py` and uncomment the `example_usage()` function call
5. Run the script: `python main.py`

## How It Works

### Embedding Process

1. The original image is transformed to the frequency domain using 2D FFT
2. The watermark is embedded by modifying magnitude values in the spectrum
3. Inverse FFT is applied to get the watermarked image

### Extraction Process

1. Both original and watermarked images are transformed to the frequency domain
2. The difference between their spectrums reveals the embedded watermark
3. The extracted watermark is normalized and thresholded

## Output

The script generates the following output in the `output` directory:

- `watermarked.png`: The image with embedded watermark
- `extracted_watermark.png`: The extracted watermark
- `results.png`: Visualization comparing all images

## Limitations

- Works best with grayscale images
- The watermark should be smaller than the original image
- The original image is required for watermark extraction

## Future Improvements

- Support for color images
- Blind watermark extraction (without original image)
- Improved robustness against image manipulations
- Parameter optimization for better watermark invisibility 