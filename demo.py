from read_EG import load_EG
import numpy
import camera_registration
from compute import compute_depth, compute_triplets
from read_EG import EG, load_EG



def main():
    EGs_file = "EGs.txt"
    calibration_file = "calibration.txt"
    triplets_file = "Triplets.txt"

    EGs, EG_matrix, ncam_all = load_EG(EGs_file, calibration_file)
    depths, EG_flag = compute_depth(EGs)

    triplets_vids, triplets_num = load_triplets(triplets_file)

    triplets = compute_triplets(triplets_vids, triplets_num, depths, EGs, EG_matrix)

    camera_registration.camera_registration(EGs, triplets, ncam_all, EG_matrix, EG_flag)
    camera_registration(EGs, triplets, ncam_all, EG_matrix, EG_flag)


