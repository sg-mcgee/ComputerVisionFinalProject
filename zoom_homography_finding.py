import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# Path to your video
video_path = 'zoom_data.mp4'  # or provide full path

# Open the video
cap = cv2.VideoCapture(video_path)

# Empty list to store frames
image_list = []

# Read frames one by one
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image_list.append(frame)

cap.release()

desc = cv2.SIFT_create()
ratio_thresh = 0.6
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

keypoints_list = []
descriptors_list = []

# Extract features from each image
for img in image_list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, descs = desc.detectAndCompute(gray, None)
    keypoints_list.append(kps)
    descriptors_list.append(descs)

match_list = []
match_xy_list = []

# # Match features between consecutive images
# for idx in range(1, len(image_list)):
#     kps1 = keypoints_list[idx - 1]
#     kps2 = keypoints_list[idx]
#     descs1 = descriptors_list[idx - 1]
#     descs2 = descriptors_list[idx]

#     knn_matches = matcher.knnMatch(descs1, descs2, 2)

#     good_matches = []
#     for m, n in knn_matches:
#         if m.distance < ratio_thresh * n.distance:
#             good_matches.append(m)

#     match_list.append(good_matches)

#     match_xy = np.array([[*kps1[m.queryIdx].pt, *kps2[m.trainIdx].pt] for m in good_matches])
#     match_xy_list.append(match_xy)

# Match features between first and current image
for idx in range(1, len(image_list)):
    kps1 = keypoints_list[0]
    kps2 = keypoints_list[idx]
    descs1 = descriptors_list[0]
    descs2 = descriptors_list[idx]

    knn_matches = matcher.knnMatch(descs1, descs2, 2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    match_list.append(good_matches)

    match_xy = np.array([[*kps1[m.queryIdx].pt, *kps2[m.trainIdx].pt] for m in good_matches])
    match_xy_list.append(match_xy)




def myFindHomography(match_xy):
# # Input
# #     match_xy: 
# #         first two columns: (x,y)-values in the original image
# #         second two columns: (x,y)-values in the target image
# # # Output
# #     h: return homography of transforming from original to target frame

#     #[x';y';1] = H[x;y;1]
#     #Find H
#     #H must be 3x3
#     #-> H = [h11,h12,h13;h21,h22,h23;h31,h32,h33]
#     #-> x' = h11x + h12y + h13
#     #   y' = h21x + h22y + h23
#     #    1 = h31x + h32y + h33
#     # Use psuedoinverse to solve
#     # Need to use pairs of input coordinates to generate matrix rows of form ([x1,y1,1,0,0,0],[0,0,0,x1,y1,1])
    A = []
    primes = []
    for pairs in match_xy:
        x = pairs[0]
        y = pairs[1]
        x_prime = pairs[2]
        y_prime = pairs[3]
        primes.append([x_prime,y_prime])
        A.append([x,y,1,0,0,0,-x_prime*x,-x_prime*y])
        A.append([0,0,0,x,y,1,-y_prime*x,-y_prime*y])
    A = np.array(A)
    primes = np.array(primes)
    AT = np.transpose(A)
    h = np.linalg.pinv(AT @ A)
    h = h @ AT
    h = h @ primes.flatten()
    #print(h.shape)
    
    H = np.append(h,1).reshape(3, 3)
    #print(H.shape)
    
    return H

H_list = []  # One homography per image pair

for i in range(1, len(image_list)):
    match_xy = match_xy_list[i - 1]  # Get the (N, 4) array for image i-1 to i
    src_pts = match_xy[:, 0:2]       # x1, y1
    dst_pts = match_xy[:, 2:4]       # x2, y2
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    H_list.append(H)
print(len(H_list))

def stitch_images(img1,img2,h1,h2,fs):
# Input    
#     img1: first image
#     img2: second image
#     h1: projective transform for img1
#     h2: projective transform for img2
#     fs: size of the output image
# Output
#     return img of size fs
    warped_img1 = cv2.warpPerspective(img1,h1,fs)
    warped_img2 = cv2.warpPerspective(img2,h2,fs)

    alpha = 0.5
    beta = 1-alpha
    output_img = cv2.addWeighted(warped_img1,alpha,warped_img2,beta,0.0)
    return output_img 

# Identity transform for the first image
h_identity = np.eye(3)

# Convert list of homographies to 3D NumPy array: shape (3, 3, N)
H_array = np.stack(H_list, axis=0).transpose(1, 2, 0)
print(f'H_array shape: {H_array.shape}')

# Plot each matrix element evolution over time
fig, axs = plt.subplots(3, 3, figsize=(12, 9))
fig.suptitle("Evolution of Homography Matrix Elements")
for i in range(3):
    for j in range(3):
        axs[i, j].plot(H_array[i, j, :])
        axs[i, j].set_title(f'H[{i},{j}]')
        axs[i, j].set_xlabel('Image Pair Index')
        axs[i, j].set_ylabel('Value')
        axs[i, j].grid(True)
plt.tight_layout()
# Filter each element of the homography matrix over time
H_filtered = np.empty_like(H_array)
for i in range(3):
    for j in range(3):
        # Median filter
        # H_filtered[i, j, :] = median_filter(H_array[i, j, :], size=3)

        # Moving average filter
        H_filtered[i, j, :] = np.convolve(H_array[i, j, :], np.ones(31)/31, mode='same')
        H_filtered[i, j, :] = np.convolve(H_filtered[i, j, :], np.ones(31)/31, mode='same')
        H_filtered[i, j, :] = np.convolve(H_filtered[i, j, :], np.ones(31)/31, mode='same')
        
        # Optional: Use Savitzky-Golay for smoother, edge-preserving results
        # H_filtered[i, j, :] = savgol_filter(H_array[i, j, :], window_length=7, polyorder=2)


# Plot each matrix element evolution over time
fig, axs = plt.subplots(3, 3, figsize=(12, 9))
fig.suptitle("Evolution of Homography Matrix Elements, Filtered")
for i in range(3):
    for j in range(3):
        axs[i, j].plot(H_filtered[i, j, :])
        axs[i, j].set_title(f'H[{i},{j}]')
        axs[i, j].set_xlabel('Image Pair Index')
        axs[i, j].set_ylabel('Value')
        axs[i, j].grid(True)
plt.tight_layout()

# Save and convert back to list of 3x3 matrices
np.save('dolly_zoom_v2.npy', H_filtered)
h_list = [H_filtered[:, :, i] for i in range(H_filtered.shape[2])]

#Modify H_filtered
time = np.linspace(0,1,len(h_list))
H_filtered[0,0,:] = 1
H_filtered[0,1,:] = (0.04) / (1 + np.exp(-10*time + 5)) #Rotation
H_filtered[0,2,:] = (-35) / (1 + np.exp(-10*time + 5))
H_filtered[1,0,:] = (-0.04) / (1 + np.exp(-10*time + 5)) #Rotation
H_filtered[1,1,:] = 1
H_filtered[1,2,:] = (-10) / (1 + np.exp(-10*time + 5))
# H_filtered[2,0,:] = H_filtered[2,0,:]
H_filtered[2,0,:] = 4*(-7.5e-5) / (1 + np.exp(-10*time + 5))
# H_filtered[2,1,:] = 10*(7.5e-5) / (1 + np.exp(-10*time + 5))
# H_filtered[2,0,:] = 0
# H_filtered[2,1,:] = H_filtered[2,1,:] 
H_filtered[2,1,:] = 0
H_filtered[2,2,:] = 1

np.save('dolly_zoom_v3.npy', H_filtered)
h_list = [H_filtered[:, :, i] for i in range(H_filtered.shape[2])]





# Plot each matrix element evolution over time
fig, axs = plt.subplots(3, 3, figsize=(12, 9))
fig.suptitle("Evolution of Homography Matrix Elements, Manually Defined")
for i in range(3):
    for j in range(3):
        axs[i, j].plot(H_filtered[i, j, :])
        axs[i, j].set_title(f'H[{i},{j}]')
        axs[i, j].set_xlabel('Image Pair Index')
        axs[i, j].set_ylabel('Value')
        axs[i, j].grid(True)
plt.tight_layout()
plt.show()

# Visual overlay of warped images using filtered homographies
current_index = 1
fs = (image_list[0].shape[1], image_list[0].shape[0])  # Width x Height

while True:
    prev_idx = (current_index - 1) % (len(image_list) - 1)
    curr_idx = current_index % len(image_list)

    img1 = image_list[prev_idx]
    img2 = image_list[curr_idx]

    h1 = h_identity
    h2 = h_list[prev_idx]  # ✅ Use filtered homography

    warped_img1 = cv2.warpPerspective(img1, h1, fs)
    warped_img2 = cv2.warpPerspective(img2, h2, fs)
    overlay = cv2.addWeighted(warped_img1, 0.5, warped_img2, 0.5, 0)

    cv2.imshow('Overlay', overlay)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        current_index = (current_index + 1) % len(image_list)
    elif key == ord('a'):
        current_index = (current_index - 1) % len(image_list)

cv2.destroyAllWindows()



# cv2.imshow('test',stitch_images(image_list[0],image_list[1],h1,h2,(image_list[0].shape[1],image_list[0].shape[0])))
# cv2.waitKey()