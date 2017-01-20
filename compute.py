from read_EG import EG, load_EG
import numpy as np
import camera_registration


class Triplet(object):
    def __init__(self, pid, vid, s_012, s_201, s_120):
        self.pid = pid
        self.vid = vid
        self.s_012 = s_012
        self.s_201 = s_201
        self.s_120 = s_120


def compute_depth(EGs):
    """

    :param EGs:
    :return:
    """
    npair = len(EGs)
    depths = []
    EG_flag = np.zeros(npair)
    for i in range(npair):
        print 'compute depth pair No.' + str(i)
        match = EGs[i].matches
        point_num = match.shape[0]
        R = EGs[i].R

        # Trangulation
        temp_good = 0
        depth = np.zeros((point_num, 2))
        for j in range(point_num):
            depth[j, :] = [-1, -1]
            p0 = np.matrix([match[j,  1], match[j, 2], 1]).transpose()
            p1 = np.matrix([match[j, 4], match[j, 5], 1]).transpose()
            plc = R.transpose()*p1
            angle = np.arccos(p0.transpose()*plc/np.linalg.norm(p0)/np.linalg.norm(plc))/np.pi*180
            if angle < 1:  # If the angle is too small it will be reduced
                continue
            A = np.hstack((p0, -plc))
            b = (-R.transpose().dot(EGs[i].t)).reshape((3, 1))
            # b = np.vstack((b, b)).transpose()
            dp = np.linalg.lstsq(A, b)[0]

            pt3d = 0.5*(p0*dp[0] + plc*dp[1]+b)

            proj0 = pt3d/pt3d[2]
            proj1 = R.dot(pt3d-b)
            proj1 = proj1/proj1[2]
            projerr = 0.5*(np.linalg.norm((proj0[0:2]) - p0[0:2]) + np.linalg.norm(proj1[0:2] - p1[0:2]))

            if projerr < 0.01:
                depth[j, :] = dp.transpose()
                temp_good += 1
        if temp_good > 0:
            EG_flag[i] = 1
        depths.append(depth)

    return depths, EG_flag


def load_triplets(triplets_file):
    """
    Load triplets from file
    :param triplets_file:
    :return: triplet object array
    """
    with open(triplets_file, 'r') as file:
        triplets_vids = np.loadtxt(file)
        triplets_num = triplets_vids.shape[0]
    return triplets_vids, triplets_num


def compute_baseline_ratio(matches1, matches2, depths1, depths2, commonIDs):
    """

    :param matches1, matches2: The feature matches of EG1 and EG2.
    :param depths1, depths2: The depths computed from EG1 and EG2, repsectively.
    :param commonIDs: The ids (1 or 2) of the common view in EG1 and EG2.
    For matches01 and matches12, the CommonIDs is [2 1].
    """
    nmatch1 = matches1.shape[0]
    nmatch2 = matches2.shape[0]

    inds = -1
    ratio12_all = np.zeros(min(nmatch1, nmatch2))
    for id1 in range(nmatch1):
        f0 = matches1[id1, commonIDs[0]*3-3]
        id2 = np.where(matches2[:, commonIDs[1]*3-3] == f0)
        if id2[0].size != 0:
            if depths1[id1, 0] > 0 and depths1[id1, 1] > 0 and depths2[id2, 0] > 0 and depths2[id2, 1] > 0:
                inds = inds + 1
                ratio12_all[inds] = depths2[id2, commonIDs[1]-1]/depths1[id1, commonIDs[0]-1]

    ratio12_all = ratio12_all[:inds+1]
    ratio12 = 0
    # May use RANSAC here to select the best one
    if inds > 1:
        id_s = max([round(0.3*inds), 1])
        id_l = min([round(0.7*inds), inds])
        ratio12_all = np.sort(ratio12_all)
        ratio12 = np.mean(ratio12_all[id_s:id_l])  # might cause problem here
    return ratio12


def compute_triplets(triplets_vids, triplets_num, depths, EGs, EG_matrix):
    triplets = []
    for i in range(triplets_num):
        pid01 = int(EG_matrix[triplets_vids[i, 0], triplets_vids[i, 1]])
        pid02 = int(EG_matrix[triplets_vids[i, 0], triplets_vids[i, 2]])
        pid12 = int(EG_matrix[triplets_vids[i, 1], triplets_vids[i, 2]])
        s_012 = compute_baseline_ratio(EGs[pid01].matches, EGs[pid12].matches, depths[pid01], depths[pid12], [2, 1])
        s_201 = compute_baseline_ratio(EGs[pid02].matches, EGs[pid01].matches, depths[pid02], depths[pid01], [1, 1])
        s_120 = compute_baseline_ratio(EGs[pid12].matches, EGs[pid02].matches, depths[pid12], depths[pid02], [2, 2])
        triplet = Triplet([pid01, pid02, pid12], triplets_vids[i, :4], s_012, s_201, s_120)
        triplets.append(triplet)
    return triplets

