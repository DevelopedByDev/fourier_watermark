import numpy as np
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt

# Naive Fourier watermarking of Text

tokens = [637, 922, 22512, 326, 945, 34460, 2101, 922, 11609, 10175, 668, 1236, 10519, 484, 9790, 1339, 21087, 1072, 306, 922, 4246, 4862, 3630]

tokensF= np.fft.fft(tokens)

filter = np.ones_like(tokensF)

def normalize(I):
    min_val = I.min()
    max_val = I.max()
    norm = (I - min_val) / (max_val - min_val)
    return (norm * I.max()).astype(int)

# print(int(filter.shape[0]/2)-1)

filter[int(filter.shape[0]/2)-1] = 0

print(filter)

outputF = tokensF*filter

outputtokens = (np.real(np.fft.ifft(outputF)))

print(outputtokens)

outputtokens=normalize(outputtokens)

print(outputtokens, outputtokens.shape)

# Fourier Watermarking of videos

def FWM(I):
    return 2*I

frames = []

path = "videoplayback.mp4"
cap = cv2.VideoCapture(path)
ret = True

while ret:
    ret, img = cap.read()  # read one frame
    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        frames.append(img)

cap.release()

video = np.stack(frames, axis=0)  # (T, H, W)

watermarked_video = np.stack([FWM(slice) for slice in video], axis=0)

# plt.imshow(watermarked_video[0, :, :], cmap=cm.gray, vmin=0, vmax=255)

from matplotlib.animation import FuncAnimation, PillowWriter

# Create figure and axis
fig, ax = plt.subplots()
watermarked_playback = ax.imshow(watermarked_video[0], cmap='gray', vmin=0, vmax=255)
ax.axis('off')  # optional: turn off axis

# Update function
def update(frame):
    watermarked_playback.set_data(watermarked_video[frame])
    return [watermarked_playback]

# Create animation
ani = FuncAnimation(fig, update, frames=range(video.shape[0]), blit=True)

# Save as MP4 or GIF
# ani.save("output_video.mp4", fps=20)  # or use PillowWriter for GIF
ani.save("output_video.gif", writer=PillowWriter(fps=15))
plt.show()
plt.close()  # close the plot window

