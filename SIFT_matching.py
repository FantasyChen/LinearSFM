# Test on SIFT_Matching by OpenCV
import sys

sys.path.insert(0, "/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python")
sys.path.insert(1, "/mnt/scratch/third-party-packages/libopencv_3.1.0/lib")

import cv2
import numpy as np


def get_calibration(filename):
    """

    :param filename:
    :return: the camera matrix contains the information of focal length and principle point
    """
    file = open(filename, 'r')
    if not file:
        return None
    line = file.readline().lstrip('P0: ')
    line = np.matrix(line.split()).reshape((3, 4))
    project_matrix = np.float64(line)
    camera_matrix = cv2.decomposeProjectionMatrix(projMatrix=project_matrix)[0]
    return camera_matrix


def get_image_path(folder_path='./', index=0, format='.png'):
    """
    :param folder_path: absolute path of the folder
    :param index: index of image
    :param format: format of image
    :return: absolute path of the image
    """
    fixed_length = 6
    fixed = (fixed_length - len(str(index))) * '0'
    image_name = fixed + str(index) + format
    return folder_path + image_name

image_folder = '/mnt/scratch/haoyiliang/KITTI/odometry/data_odometry_gray/sequences/00/image_0/'
calibration_file = '/mnt/scratch/haoyiliang/KITTI/odometry/data_odometry_gray/sequences/00/calib.txt'
image_size = 20
image_buffer = 10
R = []
t = []
sift = cv2.xfeatures2d.SIFT_create()
camera_matrix = get_calibration(calibration_file)
focal = camera_matrix[0][0]
principle = (camera_matrix[0][2], camera_matrix[1][2])


for i in range(image_size - image_buffer):
    image1 = cv2.imread(get_image_path(image_folder, i))
    print i
    for j in range(i + 1, i + image_buffer + 1):
        image2 = cv2.imread(get_image_path(image_folder, j))
        kp1, des1 = sift.detectAndCompute(image1, None)
        kp2, des2 = sift.detectAndCompute(image2, None)
        bf = cv2.BFMatcher()  # Brute-Force Matcher (FLAAN matcher is an alternative)
        matches = bf.knnMatch(des1, des2, k=2)
        # kpimg1 = cv2.drawKeypoints(image1, kp1, None)
        # kpimg2 = cv2.drawKeypoints(image2, kp2, None)
        good = []
        pts1 = []
        pts2 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append([m])
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        ess_matrix, mask = cv2.findEssentialMat(pts1, pts2, focal, principle)
        rotation1, rotation2, translation = cv2.decomposeEssentialMat(ess_matrix)
        R.append(rotation1)
        t.append(translation)

# FLANN match between images
# flann = cv2.FlannBasedMatcher()
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2, k=2)
