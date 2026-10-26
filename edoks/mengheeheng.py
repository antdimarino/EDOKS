# ------------------------------------------------------------------------
# EDOK Metric
# Authors: Vincenzo Bevilacqua, Antonio Di Marino 
# Copyright (c) 2026. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
from scipy.spatial import distance
import numpy as np
import sys
import warnings

class MengHeeHeng():

    def __init__(self):
        self.centroids = []
        self.number_cluster = 2
        self.cluster = [[], []]
        self.max_num_clusters = 100
    
    def fit(self, feature_vectors):
        """
        Function that applies the MengHee-Heng algorithm to compute clustering between n-dimensional points.

        Args:
            feature_vectors (np.ndarray): A list of n-dimensional points.
        Returns:
            centroids (python list): A list of k centroids.
            weights (python list): A list of corrispective k weights.
        """

        self.idCluster = np.zeros(feature_vectors.shape[0])
        #find the two point with maximum distance in feature vectors
        d = distance.cdist(feature_vectors, feature_vectors)
        bestpair = np.unravel_index(d.argmax(), d.shape)
        self.centroids.append(feature_vectors[bestpair[0]]) 
        self.centroids.append(feature_vectors[bestpair[1]]) 

        self.updateCluster(feature_vectors)
        condition = True
        while(condition):

            end = True

            dist_max = 0
            old_k = -1
            old_id_c = -1
            for k in range(self.number_cluster):
                for id_c, c in enumerate(self.cluster[k]):
                    dist = np.linalg.norm(np.array(self.centroids[k]) - np.array(c), ord=1)
                    if dist > dist_max:
                        dist_max = dist
                        new_ct = c
                        old_k = k
                        old_id_c = id_c

            dist_couple = []
            for k in range(self.number_cluster):
                for w in range(k+1, self.number_cluster):
                    dist_couple.append(np.linalg.norm(np.array(self.centroids[k]) - np.array(self.centroids[w]), ord=1))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                q = np.mean(dist_couple)

            if dist_max > q/4:
                end = False
                self.number_cluster += 1
                self.centroids.append(new_ct)

                #Rimuovi solo il new_ct dal vecchio cluster
                self.cluster[old_k].pop(old_id_c)
                self.centroids[old_k] = np.nan_to_num(np.mean(self.cluster[old_k], axis=0)).tolist()
             
                self.cluster = [ [] for _ in range(self.number_cluster) ]

                self.updateCluster(feature_vectors)

            if end == True:
                break

            if len(self.centroids) == self.max_num_clusters:
                condition = False

        if len(self.cluster) == 2 and len(self.cluster[1]) == 0:
            del self.cluster[1]
            del self.centroids[1]

        return self.centroids, self.cluster

    def updateCluster(self, feature_vectors):

        for idx, p in enumerate(feature_vectors):
            min_value = sys.float_info.max
            for k in range(self.number_cluster):
                dist = np.linalg.norm(self.centroids[k] - p, ord=1)
                if dist < min_value:
                    id = k
                    min_value = dist
            self.idCluster[idx] = id
            self.cluster[id].append(p.tolist())

        for k in range(self.number_cluster):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if self.cluster[k]:
                    self.centroids[k] = np.nan_to_num(np.mean(self.cluster[k], axis=0)).tolist()