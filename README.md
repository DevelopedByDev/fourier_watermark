# Fourier Transform Watermarking

A digital watermarking system using the Discrete Fourier Transform (DFT) for both images and videos.

## Overview

This project implements frequency domain watermarking by embedding data in the Fourier spectrum of images and videos for copyright protection and authenticity verification.

## Features

- Image watermarking using DFT
- Video watermarking with frame-by-frame processing
- Time-domain video watermarking
- Watermark extraction and visualization
- Spectrum analysis tools

## Requirements

```
pip install numpy opencv-python matplotlib
```

## Core Functions

### Image Processing
- **embed_watermark**: Embeds a watermark in an image using frequency domain manipulation
- **extract_watermark**: Recovers a watermark from a watermarked image
- **visualize_spectrum**: Displays the Fourier transform magnitude spectrum

### Video Processing
- **embed_video_watermark_TD**: Embeds a 1D signal watermark in a video using time-domain techniques
- **extract_video_watermark_TD**: Extracts the embedded watermark from a video
- **load_video**: Converts video files to processable NumPy arrays

### Demo Functions
- **example_usage_image**: Demonstrates the complete image watermarking pipeline
- **example_usage_video**: Shows frame-by-frame video watermarking
- **example_usage_video_TD**: Demonstrates time-domain video watermarking

## Usage

1. Create an `input` directory
2. Add your source files:
   - For images: `input/original.png` and `input/watermark.png`
   - For videos: `input/meatthezoo.mp4` and `input/youtube_watermark.jpg`
3. Run the appropriate example function in `main.py`:
   ```python
   # For image watermarking
   example_usage_image()
   
   # For video watermarking
   example_usage_video()
   
   # For time-domain video watermarking
   example_usage_video_TD()
   ```

## How It Works

### Image Watermarking
1. Transform the image to frequency domain using FFT
2. Modify magnitude values to embed the watermark
3. Apply inverse FFT to get the watermarked image
4. Extract by comparing frequency spectra of original and watermarked images

### Video Watermarking
- **Frequency Domain**: Process each frame using the image watermarking technique
- **Time Domain**: Embed a sinusoidal signal in specific pixel locations across frames

## Output

Results are saved to the `output` directory, including watermarked media and extraction visualizations.

## Limitations & Future Work

- Currently optimized for grayscale images
- Non-blind watermarking (requires original for extraction)
- Future: Support for color images, blind extraction, improved robustness