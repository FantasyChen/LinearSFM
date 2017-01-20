import numpy as np
from io import StringIO


class EG(object):
    def __init__(self, vid, R, t, matches):
        self.vid = vid
        self.R = R
        self.t = t
        self.matches = matches


def load_EG(EG_file, calib_file):  # Calibration if needed
    """

    :param EG_file:
    :param calibration: array[0] = focal length, array[1, 2] = principle point (cx, cy), array[3] = number of camera
    :return: EG object, EG pairwise matrix
    """
    with open(EG_file, 'r') as file,\
        open(calib_file, 'r') as calib:
        if not file or not calib:
            return None
        calibration = map(float, calib.readline().split())
        lines = file.readlines()
        i = 0
        pair_num = 0
        EG_list = []
        cam_num = int(calibration[3])  # Read from the calibration file to get the number of cameras
        # EG_matrix denote the pairwise index of cameras
        EG_matrix = -np.ones((cam_num, cam_num))
        while i < len(lines):
            curline = lines[i]
            label = curline[0]
            if label == 'p':
                cur = curline.split()[1:]
                point_num, camera_id_1, camera_id_2= map(int, cur)
                i += 1
                R = np.loadtxt(StringIO(unicode("".join(lines[i:i+3]))))
                vid = [camera_id_1, camera_id_2]
                t = np.loadtxt(StringIO(unicode(lines[i+3])))
                # Raw matched points
                matches = np.loadtxt(StringIO(unicode("".join(lines[i+4:i+4+point_num]))))
                # Normalize image coordinates
                matches = np.vstack((matches[:, 0], (matches[:, 1]-calibration[1])/calibration[0],
                                     (matches[:, 2]-calibration[2])/calibration[0],
                                    matches[:, 3], (matches[:, 4]-calibration[1])/calibration[0],
                                     (matches[:, 5]-calibration[2])/calibration[0]))
                EG_matrix[camera_id_1, camera_id_2] = pair_num
                # print matches
                curEG = EG(vid, R, t, matches.transpose())
                # print curEG.matches
                EG_list.append(curEG)
                i = i + 4 + point_num
                pair_num += 1
            else:
                i += 1
        return EG_list, EG_matrix, cam_num

# EG_list, EG_matrix = load_EG('./EGs.txt', './calibration.txt')
