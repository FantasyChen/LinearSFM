import numpy as np
import scipy.sparse as sparse
import numpy.matlib as matlib
import scipy.sparse.linalg


def camera_registration(EGs, triplets, ncam_all, EG_matrix, EG_flag):
    """

    :param EGs:The object array with the information of pair-wise epipolar geometry.
    :param triplets: The object array with the triplet information.
    :param ncam_all: The number of all cameras
    :param EG_matrix: The index matrix of EGs
    :param EG_flag: The flag for each EG. (0 --  Bad EG; 1 -- Good EG.)
    """
    graph = EG_matrix
    graph = graph + 1
    for i in range(len(EG_flag)):
        if not EG_flag[i]:
            graph[EGs[i].vid[0], EGs[i].vid[1]] = 0

    graph = graph.T + graph
    camid = find_largest_connected_componet(graph)
    # Rotation registration
    EGs = [EGs[i] for i, v in enumerate(EG_flag) if v == 1]
    R = rotation_registration(EGs, camid, ncam_all)  # TODO

    # find the largest connected cluster among all good triplets
    triplet_num = len(triplets)
    triplet_graph = np.zeros(triplet_num)
    triplet_flag = np.zeros(triplet_num)
    for i in range(triplet_num):
        pid01 = triplets[i].pid[0]
        pid02 = triplets[i].pid[1]
        pid12 = triplets[i].pid[2]
        if EG_flag[pid01] and EG_flag[pid02] and EG_flag[pid12]:
            if triplets[i].s_012 > 0 and triplets[i].s_201 > 0 and triplets[i].s_120 > 0:
                triplet_flag[i] = 1

    # Compute the global translation

    # Find the largest connected triplet group
    triplet_validIDs = np.transpose(np.nonzero(triplet_flag))
    pids = np.zeros((3, len(triplet_validIDs)))
    for i in range(len(triplet_validIDs)):
        cols1 = np.transpose(np.nonzero(pids == pids[0, i]))[:, 1:2]
        cols2 = np.transpose(np.nonzero(pids == pids[1, i]))[:, 1:2]
        cols3 = np.transpose(np.nonzero(pids == pids[2, i]))[:, 1:2]
        for j in range(3):
            triplet_graph[triplet_avlidIDs[i], triplet_validIDs[np.matrix[[cols0], [cols[2]], [cols[1]]]] = -1
            triplet_graph[triplet_validIDs[i], triplet_validIDs[np.matrix[[cols1], [cols2], [cols3]]]] = 1

            triplet_graph = triplet_graph + triplet_graph.transpose()
            tripletid = find_largest_connected_componet(triplet_graph)

            cam_w = np.zeros(ncam_all)
            EG_flag = np.zeros(len(EG_flag))  # TODO
            for i in range(tripletid):
                cam_w[triplets[tripletid[i]].vid] = cam_w[triplets[tripletid[i]].vid]
            EG_flag[triplets[tripletid[i]].pid] = 1
            camid = cam_w.ravel().nonzero()
            # Translation registration
            C = translation_registration(EGs, EG_flag, triplets[tripletid], R, camid, cam_w)


def rotation_registration(EGs, camid, ncam_all):
    """

    :param EGs: The structure array with the information of pair-wise EG.
    :param camid: The ID of the cameras.
    :param ncam_all:
    """
    npair = len(EGs)
    ncam = len(camid)
    R = np.zeros(shape=(3, 3, ncam_all))
    A = np.zeros(shape=(npair * 3, ncam * 3))
    for i in range(npair):
        v0 = EGs[i].vid[0]
        v1 = EGs[i].vid[1]
        r = 3 * i
        c0 = np.transpose(np.where(camid == v0))
        c0 = c0[0][0]
        c1 = np.transpose(np.where(camid == v1))
        c1 = c1[0][0]
        w = 1.0
        for j in range(3):
            A[r + j, c1 + j + 1] = w
            for k in range(3):
                A[r + j, c0 + k + 1] = -EGs[i].R[j, k] * w
    A = sparse.csc_matrix(A)
    ApA = sparse.csc_matrix(A.T.dot(A))
    D, V = sparse.linalg.eigsh(ApA, k=3, sigma=-1e-6)  # Check different with matlab
    Rc = np.zeros(shape=(3, 3, ncam))
    for i in range(ncam):
        R0 = V[3 * i:3 * (i + 1), 0:3]
        u, s, v = np.linalg.svd(R0)
        Rc[:, :, i] = u.dot(v)
    dR = Rc[:, :, 0]

    for i in range(ncam):
        R[:, :, camid[i][0]] = Rc[:, :, i].dot(dR.transpose())
    # for i in range(ncam):
    #         print R[:, :, i]
    return R


def translation_registration(EGs, EG_flag, triplets, R, camid, cam_w):
    """

    :param EGs: The structure array with the information of pair-wise EG.
    :param EG_flag: The flag of each EG.
    :param Triplets: The object array with triplet information.
    :param R: The global rotation matrix for all cameras.
    :param camid: The ID of the connected cameras.
    :param cam_w: The occurence number of all camera in the largest connected group.
    """
    C = np.zeros(3, len(cam_w))
    ncam = len(camid)
    triplet_num = len(triplets)
    EG_num = len(EGs)
    A = np.zeros(triplet_num * 9, ncam * 3)
    for tri_index in range(triplet_num):
        triplet_temp = triplets[tri_index]
        c0 = triplet_temp.vid[0]
        c1 = triplet_temp.vid[1]
        c2 = triplet_temp.vid[2]

        w = np.sqrt(1 / min(cam_w[triplet_temp.vid]))
        t01 = -np.dot(R[:, :, c1].T, EGs[triplet_temp.pid[0]].t)
        t02 = -np.dot(R[:, :, c2].T, EGs[triplet_temp.pid[1]].t)
        t12 = -np.dot(R[:, :, c2].T, EGs[triplet_temp.pid[2]].t)

        # Rotation from t12 to t10
        R120 = compute_axis_rotation(t12, -t01)
        # Rotation from t01 to t02
        R201 = compute_axis_rotation(t01, t02)
        # Rotation from t20 to t21
        R120 = compute_axis_rotation(-t02, -t12)

        s_012 = triplet_temp.s_012
        s_201 = triplet_temp.s_201
        s_120 = triplet_temp.s_120

        eq_num = (tri_index - 1) * 9
        c0 = 3 * (np.transpose(np.nonzero(camid == c0) - 1))  # CHECK
        c1 = 3 * (np.transpose(np.nonzero(camid == c1) - 1))
        c2 = 3 * (np.transpose(np.nonzero(camid == c2) - 1))

        # equation (5) in the paper
        A[eq_num:eq_num + 3, c0:c0 + 3] = (-s_201.dot(R201) + R012.T / s_012 + np.eye(3)) * w
        A[eq_num:eq_num + 3, c1:c1 + 3] = (s_201.dot(R201) - R012.T / s_012 + np.eye(3)) * w
        A[eq_num:eq_num + 3, c0:c0 + 3] = -2 * np.eye(3) * w

        # equation (6) in the paper
        A[eq_num + 3:eq_num + 6, c0:c0 + 3] = (-R201.T / s_201 + s_120 * R120 + np.eye(3)) * w
        A[eq_num + 3:eq_num + 6, c1:c1 + 3] = -2 * np.eye(3) * w
        A[eq_num + 3:eq_num + 6, c2:c2 + 3] = (R201.T / s_201 - s_120 * R120 + np.eye(3)) * w

        # equation (7) in the paper
        A[eq_num + 6:eq_num + 9, c0:c0 + 3] = - 2 * np.eye(3) * w
        A[eq_num + 6:eq_num + 9, c1:c1 + 3] = (-s_012 * R012 + R120.T / s_120 + np.eye(3)) * w
        A[eq_num + 6:eq_num + 9, c2:c2 + 3] = (s_012 * R012 - R120.T / s_120 + np.eye(3)) * w

    ApA = A.T.dot(A)

    D, V = sparse.linalg.eigsh(ApA, k=4, sigma=-1e-6)  # CHECK

    maxid = np.argmax(D.max(1))

    Cc = V[:, maxid:maxid + 1].reshape(3, -1, order='F').copy()
    Cc = Cc - matlib.repmat(Cc[:, 0:1], 1, Cc.shape[1])

    _, S, _ = np.linalg.svd(Cc)

    # Check whether all the cameras are in one plane or a line
    if S[2, 2] < 1e-6:  # rank(Cc) < 3
        temp_t = np.zeros((3, EG_num))
        for i in range(EG_num):
            temp_t[:, i] = (Cc[:, np.transpose(np.nonzero(camid == EGs[i].vid[1]))]
                            - Cc[:, np.transpose(np.nonzero(camid == EGs[i].vid[0]))])
            temp_t[:, i] = temp_t[:, i] / np.linalg.norm(temp_t[:, i:i + 1])

        if S[1, 1] > 1e-6:  # rank(Cc) < 2
            axis = np.cross(temp_t[:, 0:1], temp_t[:, 2:3])
            axis = axis / np.linalg.norm(axis)

            theta = np.zeros(EG_num, 1)
            for i in range(EG_num):
                if EG_flag[i]:
                    rfc = -R[:, :, EGs[i].vid[1]].T.dot(EGs[i].t)

                    # Projection
                    rfc_p = rfc - (rfc.T.dot(axis)).dot(axis)
                    theta[i] = np.arccos(temp_t[:, i:i + 1].T.dot(rfc_p / np.linalg.norm(rfc_p)))

                    if np.cross(temp_t[:, i:i + 1], rfc_p).T.dot(axis) < 0:
                        theta[i] = 2 * np.pi - theta[i]
            # Compute the mean value
            temp = theta - theta[0]
            tempid = np.transpose(np.nonzero(np.abs(temp) > np.pi))
            if tempid.size != 0:
                for k in range(len(tempid)):  # may change to .shape
                    if temp[tempid[k]] > 0:
                        theta[tempid[k]] = theta[tempid[k]] - 2 * np.pi
                    else:
                        theta[tempid[k]] = theta[tempid[k]] + 2 * np.pi
            thetaM = np.mean(theta)

            # From axis-angle to rotation matrix
            nJ = np.matrix([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            dR = np.eye(3) + nJ.dot(np.sin(thetaM)) + nJ.dot(nJ).dot(1 - np.cos(thetaM))
        else:
            # The cameras are considered on the same line.
            rf_t = zeros(3, EG_num)
            for i in range(EG_num):
                if EG_flag[i]:
                    rf_t[:, i] = -R[:, :, EG[i].vid[1]].T.dot(EGs[i].t)
                    rf_t[:, i] = rf_t[:, i] / np.linalg.norm(rf_t[:, i])

            rfdm = np.mean(rf_t, axis=1)
            tempdn = temp_t[:, 0]

            axis = np.cross(tempdm, rfdm)

            if np.linalg.norm(axis) != 0:
                axis = axis / np.linalg.norm(axis)
                thetaM = np.arccos(tempdm.T.dot(rfdm) / np.linalg.norm(tempdm) / np.linalg.norm(rfdm))
                nJ = np.matrix([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                dR = np.eye(3) + nJ.dot(np.sin(thetaM)) + nJ.dot(nJ).dot(1 - np.cos(thetaM))
            else:
                if rfdm.T.dot(tempdm) > 0:  # Check
                    dR = np.eye(3)
                else:
                    dR = -np.eye(3)
    else:
        pos_num = 0
        neg_num = 0

        for i in range(EG_num):
            if EG_flag[i]:
                temp_c = (Cc[:, np.transpose(np.nonzero(camid == EG[i].vid[1]))] -
                          Cc[:, np.transpose(np.nonzero(camid == EG[i].vid[0]))])
                rfc = -R[:, :, EGs[i].vid[1]].T.dot(EGs[i].t)

                if temp_c.T.dot(rfc) > 0:
                    pos_num += 1
                else:
                    neg_num -= 1
        if pos_num > neg_num:
            dR = np.eye(3)
        else:
            dR = -np.eye(3)
    for i in range(ncam):
        C[:, camid[i]] = dR.dot(Cc[:, i])
    return C


def find_largest_connected_componet(graph):
    """
    Need to translated
    :param graph:
    """
    S, C = sparse.csgraph.connected_components(graph)
    cluster_num = np.zeros(S)
    for i in range(S):
        cluster_num[i] = len(np.transpose(np.nonzero(C == i)))
    max_cluster_id = np.argmax(cluster_num)
    largestid = np.transpose(np.nonzero(C == max_cluster_id))
    return largestid

def compute_axis_rotation(t01, t02):
    R201 = np.zeros((3, 3))
    cos201 = min(max(t02.T.dot(t01)[0], -1.0), 1.0)
    sin201 = np.sqrt(1 - cos201 * cos201)

    if sin201 > 0.002:
        # print t01, t02
        axis = np.cross(t01, t02, axis=0)
        axis = axis / np.linalg.norm(axis)
        u = axis[0]
        v = axis[1]
        w = axis[2]

        R201[0, 0] = u * u + (v * v + w * w) * cos201
        R201[1, 0] = u * v * (1 - cos201) + w * sin201
        R201[2, 0] = u * w * (1 - cos201) - v * sin201

        R201[0, 1] = u * v * (1 - cos201) - w * sin201
        R201[1, 1] = v * v + (u * u + w * w) * cos201
        R201[2, 1] = v * w * (1 - cos201) + u * sin201

        R201[0, 2] = u * w * (1 - cos201) + v * sin201
        R201[1, 2] = v * w * (1 - cos201) - u * sin201
        R201[2, 2] = w * w + (u * u + v * v) * cos201

    else:
        if cos201 > 0:
            for i in range(3):
                R201[i, i] = 1
        else:
            for i in range(3):
                R201[i, i] = -1
    return R201
