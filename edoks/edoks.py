# ------------------------------------------------------------------------
# EDOK Metric
# Authors: Vincenzo Bevilacqua, Antonio Di Marino 
# Copyright (c) 2026. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
from .emd import EMD
from .oklab import Oklab as OK
import sys
import cv2
import os
import numpy as np
import warnings

class EDOKS():
    def __init__(self, freq=[0.1, 0.2, 0.3, 0.4], n_or=6, p_size=128):
        self.ok = OK()
        self.emd = EMD(frequencies=freq, num_orientations=n_or, patch_size=p_size)
        self.ok_score = None
        self.emd_score = None
        self.edok_score = None
        self.edoks_score = None
    
    def edok(self):
        """
        Function that compute EDOK score.
        """
        self.edok_score = (self.ok_score + self.emd_score)/2

    def edoks(self):
        """
        Function that compute EDOKS score.
        """
        self.edoks_score = 1/(self.edok_score + sys.float_info.min)

    def set_img1(self, img1):
        """
        Function that convert in Oklab space and extract signature from img1.
        """
        self.ok.set_img1(img1)
        self.emd.set_img1(img1)

    def compute_one_img(self, img2, name='', path_save = None):
        """
        Function that convert in Oklab space and extract signature on img2 and compute emd
        distance, color distance, edok and edoks between img1 and img2. 
        It is only useful if img1 is always the same while img2 can change.

        Args:
            img2 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns,
                            to convert in Grayscale for signature extraction.
                            Required an image with values in the range [0, 255]. 
            name (str): base name of the interpretability maps to be saved.
            path_save (str): directory where to save interpretability maps. 
                            Enter '' if you want to save to the current directory. 
                            Default = None, so maps are not saved. 
        Returns:
            EDOKS (float): The EDOKS score img1 and img2.
            EMD (float): The EMD score between textures of img1 and img2.
            OK (float): The OK score between colors of img1 and img2.
        """
        self.ok_score = self.ok.compute_one_img(img2)
        self.emd_score = self.emd.compute_one_img(img2)
        self.edok()
        self.edoks()
        if path_save is not None:
            self.save_maps(name, path_save)
    
        return self.edoks_score, self.emd_score, self.ok_score

    def compute_two_img(self, img1, img2, name='', path_save = None):
        """
        Function that convert in Oklab space and extract signature on img1 and img2 and 
        compute emd distance, color distance, edok and edoks between img1 and img2.
        It is useful if img1 is not always the same and changes as img2.

        Args:
            img1 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns, 
                            to convert in Grayscale for signature extraction.
                            Required an image with values in the range [0, 255].
            img2 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns, 
                            to convert in Grayscale for signature extraction.
                            Required an image with values in the range [0, 255]. 
            name (str): base name of the interpretability maps to be saved.
            path_save (str): directory where to save interpretability maps. 
                            Enter '' if you want to save to the current directory. 
                            Default = None, so maps are not saved. 
        Returns:
            EDOKS (float): The EDOKS score img1 and img2.
            EMD (float): The EMD score between textures of img1 and img2.
            OK (float): The OK score between colors of img1 and img2.
        """
        self.ok_score = self.ok.compute_two_img(img1, img2)
        self.emd_score = self.emd.compute_two_img(img1, img2)
        self.edok()
        self.edoks()
        if path_save is not None:
            self.save_maps(name, path_save)
        return self.edoks_score, self.emd_score, self.ok_score

    def save_maps(self, name, path_save):
        """
        Function for saving the three attention maps related to the terms OK, EMD, and EDOKS.

        Args:
            name (str): base name of the interpretability maps to be saved.
            path_save (str): directory where to save interpretability maps. 
                            Enter '' if you want to save to the current directory. 
        """

        if path_save is None:
            raise ValueError("the save path has not been configured.")
        
        if self.ok.diff_map is None:
            raise ValueError("the OK term has not been calculated. Use the compute_one_img or compute_two_img methods before saving the maps.")

        if self.emd.mgn_diffs_save is None:
            raise ValueError("the EMD term has not been calculated. Use the compute_one_img or compute_two_img methods before saving the maps.")

        filename, _ = os.path.splitext(os.path.basename(name))

        diff_map_norm = cv2.normalize(self.ok.diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        img_bgr = cv2.applyColorMap(diff_map_norm, cv2.COLORMAP_JET)

        ok_path_save = os.path.join(path_save, "OK_"+ filename +"_heatmap.png")
        if os.path.isfile(ok_path_save):
            warnings.warn('The map with path '+ ok_path_save +' already exists and will be overwritten. If you want to avoid overwriting, configure the name parameter differently.', UserWarning)
        
        cv2.imwrite(ok_path_save, img_bgr)

        emd_path_save = os.path.join(path_save, "EMD_"+ filename +"_heatmap.png")
        if os.path.isfile(emd_path_save):
            warnings.warn('The map with path '+ emd_path_save +' already exists and will be overwritten. If you want to avoid overwriting, configure the name parameter differently.', UserWarning)
        
        cv2.imwrite(emd_path_save, self.emd.mgn_diffs_save)
        
        mgn_diffs_save = cv2.resize(self.emd.mgn_diffs_save, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        mask = mgn_diffs_save > 0

        edges_color = np.zeros_like(img_bgr)
        edges_color[..., 0] = mgn_diffs_save
        edges_color[..., 1] = mgn_diffs_save
        edges_color[..., 2] = mgn_diffs_save

        alpha = 0.7
        result = img_bgr.copy()
        result[mask] = (
            img_bgr[mask] * (1 - alpha) +
            edges_color[mask] * alpha
        ).astype(np.uint8)


        edoks_path_save = os.path.join(path_save, "EDOKS_"+ filename +"_heatmap.png")
        if os.path.isfile(edoks_path_save):
            warnings.warn('The map with path '+ edoks_path_save +' already exists and will be overwritten. If you want to avoid overwriting, configure the name parameter differently.', UserWarning)
        
        cv2.imwrite(edoks_path_save, result)





