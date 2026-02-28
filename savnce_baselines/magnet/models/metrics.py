# Implements the localization and detection metrics proposed in [1] with extensions to support multi-instance of the same class from [2].
#
# [1] Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
# [2] Overview and Evaluation of Sound Event Localization and Detection in DCASE 2019
# Politis, Archontis, Annamaria Mesaros, Sharath Adavanne, Toni Heittola, and Tuomas Virtanen.
# IEEE/ACM Transactions on Audio, Speech, and Language Processing (2020).
#
# This script has MIT license
#

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).


import numpy as np
import torch
eps = np.finfo(float).eps
from scipy.optimize import linear_sum_assignment


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x).cpu().numpy()
        else:
            return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array([x])
    
def to_real_distance(distance, min_distance: int, max_distance: int): 
    distance = distance * (max_distance - min_distance) + min_distance
    return distance


class BaseMetrics:
    """
    Base class for all metrics.
    """
    
    def __init__(self):
        self._doa_thd = 20 # degrees
        self._doa_thd_rad = self._doa_thd * np.pi / 180 # radians
        self._sed_thd = 0.5 # ratio
        self._dist_thd = 0.1 # percentage
        self._num_classes = 21

    def cart2sph(self, x, y, z, deg_format=False):
        r = np.clip(np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-10, -1.0, 1.0)
        ele = np.arcsin(z / r)  # -pi/2->pi/2
        azi = np.arctan2(y, x)  # 0->pi, -pi->0
        if deg_format:
            ele = ele * 180 / np.pi
            azi = azi * 180 / np.pi
        return ele, azi


class SELDMetrics(BaseMetrics):
    def __init__(self, average='micro'):
        super().__init__()
        
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.

        :param num_classes: Number of sound classes.
        :param doa_thresh: DOA threshold for location sensitive detection.
        :param dist_thresh: Relative distance threshold for distance estimation
        '''
        self.eval_dist = True

        # Variables for Location-senstive detection performance
        self._TP = np.zeros(self._num_classes)
        self._FP = np.zeros(self._num_classes)
        self._FP_spatial = np.zeros(self._num_classes)
        self._FN = np.zeros(self._num_classes)

        self._Nref = np.zeros(self._num_classes)

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_AngE = np.zeros(self._num_classes)
        self._total_DistE = np.zeros(self._num_classes)

        self._DE_TP = np.zeros(self._num_classes)
        self._DE_FP = np.zeros(self._num_classes)
        self._DE_FN = np.zeros(self._num_classes)
        
        self._average = average

    def reset(self):
        self._TP = np.zeros(self._num_classes)
        self._FP = np.zeros(self._num_classes)
        self._FP_spatial = np.zeros(self._num_classes)
        self._FN = np.zeros(self._num_classes)
        self._Nref = np.zeros(self._num_classes)
        self._S = 0
        self._D = 0
        self._I = 0
        self._total_AngE = np.zeros(self._num_classes)
        self._total_DistE = np.zeros(self._num_classes)
        self._DE_TP = np.zeros(self._num_classes)
        self._DE_FP = np.zeros(self._num_classes)
        self._DE_FN = np.zeros(self._num_classes)

    def early_stopping_metric(self, _er, _f, _ae, _lr, _rde):
        """
        Compute early stopping metric from sed, doa, and dist errors.
        """
        seld_metric = np.nanmean([_er, 1 - _f, _ae / 180, 1 - _lr, _rde], 0)

        return seld_metric

    def compute_metric(self):
        '''
        Collect the final SELD scores
        ISDR is not returned and hasn't been tested

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''
        ER = (self._S + self._D + self._I) / (self._Nref.sum() + eps)
        ER = np.clip(ER, 0.0, 10.0)
        # classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            AngE = self._total_AngE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else 180.0
            DistE = self._total_DistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else 1.0
            LR = self._DE_TP.sum() / (eps + self._DE_TP.sum() + self._DE_FN.sum())

            SELD_score = self.early_stopping_metric(ER, F, AngE, LR, DistE)

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            AngE = self._total_AngE / (self._DE_TP + eps)
            AngE[self._DE_TP==0] = np.NaN
            DistE = self._total_DistE / (self._DE_TP + eps)
            DistE[self._DE_TP==0] = np.NaN
            LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)

            SELD_score = self.early_stopping_metric(np.repeat(ER, self._num_classes), F, AngE, LR, DistE)

            # classwise_results = np.array(
            #     [np.repeat(ER, self._num_classes), F, AngE, DistE, LR, SELD_score] if self.eval_dist else [
            #         np.repeat(ER, self._num_classes), F, AngE, LR, SELD_score])

            non_zero_F_indices = np.where(np.round(F,2) != 0)[0]

            F = F.mean()
            if len(non_zero_F_indices) > 0:
                AngE = np.nanmean(AngE[non_zero_F_indices])
                LR = LR[non_zero_F_indices].mean()
                SELD_score = SELD_score[non_zero_F_indices].mean()
                DistE = np.nanmean(DistE[non_zero_F_indices])
            else:
                AngE = 180.0
                LR = 0.0
                SELD_score = 1.0
                DistE = 1.0
        self.reset()
        return {'ER': ER, 'F': F, 'AngE': AngE, 'DistE': DistE, 'LR': LR, 'SELD_score': SELD_score}

    def update_metric(self, pred, labels, label_format='ACCDDOA', eval_dist=True):
        '''
        Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
        Adds the multitrack extensions proposed in paper [2]

        The input pred/gt must be Cartesian coordinates

        :param pred: tensor with shape [batch_size, time_steps, num_track*num_axis*num_class=3*4*21], 
            convert to dictionary containing the predictions for every frame,
            pred[frame-index][class-index][track-index] = [x, y, z, (distance)]
        :param labels : tensor with shape [batch_size, time_steps, max_num_goals, 4], 
        :param masks: tensor with shape [batch_size, time_steps, max_num_goals]
            convert to dictionary containing the groundtruth for every frame,
            gt[frame-index][class-index][track-index] = [x, y, z, (distance)]
        :param eval_dist: boolean, if True, the distance estimation is also evaluated
        '''
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
            labels = labels.unsqueeze(0)
        batch_size = pred.shape[0]
        for batch_cnt in range(batch_size):
            if label_format == 'ACCDDOA':
                pred_batch = self.accddoa_format_to_dcase_format(pred[batch_cnt])
                gt_batch = self.accddoa_format_to_dcase_format(labels[batch_cnt])
            else:
                raise ValueError(f"Invalid label format: {label_format}")
            pred_batch = self.organize_labels(pred_batch)
            gt_batch = self.organize_labels(gt_batch)

            assignations = [{} for i in range(self._num_classes)]
            assignations_pre = [{} for i in range(self._num_classes)]
            for frame_cnt in range(len(gt_batch.keys())):
                loc_FN, loc_FP = 0, 0
                for class_cnt in range(self._num_classes):
                    # Counting the number of referece tracks for each class
                    nb_gt_doas = len(gt_batch[frame_cnt][class_cnt]) if class_cnt in gt_batch[frame_cnt] else None
                    nb_pred_doas = len(pred_batch[frame_cnt][class_cnt]) if class_cnt in pred_batch[frame_cnt] else None
                    if nb_gt_doas is not None:
                        self._Nref[class_cnt] += nb_gt_doas
                    if class_cnt in gt_batch[frame_cnt] and class_cnt in pred_batch[frame_cnt]:
                        # True positives or False positive case

                        # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                        # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                        # the associated reference-predicted tracks.

                        # Reference and predicted track matching

                        gt_doas = np.array(list(gt_batch[frame_cnt][class_cnt].values()))
                        gt_ids = np.array(list(gt_batch[frame_cnt][class_cnt].keys()))
                        pred_doas = np.array(list(pred_batch[frame_cnt][class_cnt].values()))
                        pred_ids = np.array(list(pred_batch[frame_cnt][class_cnt].keys()))

                        # Extract distance
                        if gt_doas.shape[-1] == 4:
                            gt_dist = gt_doas[:, 3] if eval_dist else None
                            gt_doas = gt_doas[:, :3]
                        else:
                            assert not eval_dist, 'Distance evaluation was requested but the ground-truth distance was not provided.'
                            gt_dist = None
                        if pred_doas.shape[-1] == 4:
                            pred_dist = pred_doas[:, 3] if eval_dist else None
                            pred_doas = pred_doas[:, :3]
                        else:
                            assert not eval_dist, 'Distance evaluation was requested but the predicted distance was not provided.'
                            pred_dist = None

                        doa_err_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas, gt_dist, pred_dist)
                        assignations[class_cnt] = {gt_ids[row_inds[i]] : pred_ids[col_inds[i]] for i in range(len(doa_err_list))}
                        if eval_dist:
                            dist_err_list = np.abs(gt_dist[row_inds] - pred_dist[col_inds])

                        # https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#evaluation
                        Pc = len(pred_doas)
                        Rc = len(gt_doas)
                        FNc = max(0, Rc - Pc)
                        FPcinf = max(0, Pc - Rc)
                        Kc = min(Pc, Rc)
                        TPc = Kc
                        Lc = np.sum(doa_err_list > self._doa_thd or (eval_dist and dist_err_list > self._dist_thd))
                        FPct = Lc
                        FPc = FPcinf + FPct
                        TPct = Kc - FPct
                        assert Pc == TPct + FPc
                        assert Rc == TPct + FPct + FNc

                        self._total_AngE[class_cnt] += doa_err_list.sum()
                        self._total_DistE[class_cnt] += dist_err_list.sum() if eval_dist else 0

                        self._TP[class_cnt] += TPct
                        self._DE_TP[class_cnt] += TPc

                        self._FP[class_cnt] += FPcinf
                        self._DE_FP[class_cnt] += FPcinf
                        self._FP_spatial[class_cnt] += FPct
                        loc_FP += FPc

                        self._FN[class_cnt] += FNc
                        self._DE_FN[class_cnt] += FNc
                        loc_FN += FNc

                        assignations_pre[class_cnt] = assignations[class_cnt]

                    elif class_cnt in gt_batch[frame_cnt] and class_cnt not in pred_batch[frame_cnt]:
                        # False negative
                        loc_FN += nb_gt_doas
                        self._FN[class_cnt] += nb_gt_doas
                        self._DE_FN[class_cnt] += nb_gt_doas
                        assignations_pre[class_cnt] = {}
                    elif class_cnt not in gt_batch[frame_cnt] and class_cnt in pred_batch[frame_cnt]:
                        # False positive
                        loc_FP += nb_pred_doas
                        self._FP[class_cnt] += nb_pred_doas
                        self._DE_FP[class_cnt] += nb_pred_doas
                        assignations_pre[class_cnt] = {}
                    else:
                        # True negative
                        assignations_pre[class_cnt] = {}

                self._S += np.minimum(loc_FP, loc_FN)
                self._D += np.maximum(0, loc_FN - loc_FP)
                self._I += np.maximum(0, loc_FP - loc_FN)
        return

    def accddoa_format_to_dcase_format(self, accddoa):
        if isinstance(accddoa, torch.Tensor):
            accddoa = accddoa.cpu().numpy()
        if accddoa.ndim >2:
            accddoa = accddoa.reshape(-1, accddoa.shape[-1])
        
        x = accddoa[..., : self._num_classes]
        y = accddoa[..., self._num_classes : 2 * self._num_classes]
        z = accddoa[..., 2 * self._num_classes : 3 * self._num_classes]
        dist = accddoa[..., 3 * self._num_classes : 4 * self._num_classes]
        sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > self._sed_thd
        
        dcase_result_dict = {}
        for frame_cnt in range(sed.shape[0]):
            for class_cnt in range(sed.shape[1]):
                if sed[frame_cnt][class_cnt]:
                    if frame_cnt not in dcase_result_dict:
                        dcase_result_dict[frame_cnt] = []
                    dcase_result_dict[frame_cnt].append([class_cnt, 0,
                                                        x[frame_cnt][class_cnt], y[frame_cnt][class_cnt], z[frame_cnt][class_cnt], 
                                                        dist[frame_cnt][class_cnt]]) 
        return dcase_result_dict

    def organize_labels(self, _pred_dict, _max_frames = 2000):
        '''
            Collects class-wise sound event location information in every frame, similar to segment_labels but at frame level
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each frame
                dictionary_name[frame-index][class-index][track-index] = [azimuth, elevation, (distance)]
        '''
        nb_frames = _max_frames
        dcase_result_dict = {x: {} for x in range(nb_frames)}
        for frame_idx in range(0, _max_frames):
            if frame_idx not in _pred_dict:
                continue
            for [class_idx, track_idx, x, y, z, dist] in _pred_dict[frame_idx]:
                if class_idx not in dcase_result_dict[frame_idx]:
                    dcase_result_dict[frame_idx][class_idx] = {}
                dcase_result_dict[frame_idx][class_idx][track_idx] = [x, y, z, dist]

        return dcase_result_dict


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist

def distance_3d_between_doas(x1, y1, z1, x2, y2, z2, dist1, dist2):
    """
    3D distance between two cartesian DOAs with their respective distances
    :return: 3D distance in meters
    """
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    x1, y1, z1 = x1/N1 * dist1, y1/N1 * dist1, z1/N1 * dist1
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x2, y2, z2 = x2/N2 * dist2, y2/N2 * dist2, z2/N2 * dist2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def least_distance_between_gt_pred(gt_list, pred_list, gt_dist=None, pred_dist=None,
                                   opt_3d_dist=False, ret_3d_dist=False):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth DOA in Cartesian coordinates
        :param pred_list_xyz: list of predicted DOA in Carteisan coordinates
        :param gt_dist: list of ground-truth distances in meters (optional, for distance evaluation)
        :param pred_dist: list of predicted distances in meters (optional, for distance evaluation)
        :param opt_3d_dist: boolean, if True, the 3D distance is used for matching the predicted and groundtruth DOAs
        :param ret_3d_dist: boolean, if True, the 3D distance [meters] is returned instead of angular distance [degrees]
        :return: cost - distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """
    if opt_3d_dist or ret_3d_dist:
        assert gt_dist is not None and pred_dist is not None, 'Distance information is needed to compute 3D distances.'

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))
    dist_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
        if opt_3d_dist or ret_3d_dist:
            dist1 = gt_dist[ind_pairs[:, 0]]
            dist2 = pred_dist[ind_pairs[:, 1]]
            distances_3d = distance_3d_between_doas(x1, y1, z1, x2, y2, z2, dist1, dist2)
            if opt_3d_dist:
                cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_3d
            if ret_3d_dist:
                dist_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_3d
        if not (opt_3d_dist and ret_3d_dist):
            distances_ang = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
            if not opt_3d_dist:
                cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_ang
            if not ret_3d_dist:
                dist_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_ang

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = dist_mat[row_ind, col_ind]
    return cost, row_ind, col_ind
