import numpy as np

def hole_filling(img, kernel=3):
    N, M = img.shape
    for i in range(N):
        for j in range(M):
            if img[i, j] == 0:
                neighbour = img[max(int((i-(kernel-1)/2)), 0):min(int((i+(kernel-1)/2)), N), max(int((j-(kernel-1)/2)),0):min(int((j+(kernel-1)/2)), M)]
                if len(neighbour) == 0:
                    continue
                else:
                    max_val = np.amax(neighbour)
                    img[i, j] = max_val
    return img

