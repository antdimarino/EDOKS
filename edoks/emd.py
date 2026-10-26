# ------------------------------------------------------------------------
# EDOK Metric
# Authors: Vincenzo Bevilacqua, Antonio Di Marino
# Copyright (c) 2026. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
import numpy as np
import cv2
from skimage.filters import gabor
from skimage.filters import gabor_kernel
from .mengheeheng import MengHeeHeng
import matplotlib.pyplot as plt
import ctypes
import os
import sys
import math
import torch
import torch.nn.functional as F
import time
class Feature(ctypes.Structure):
    _fields_ = [("centroid", ctypes.c_float * 24)]


class Flow(ctypes.Structure):
    _fields_ = [
        ("from_", ctypes.c_int),
        ("to", ctypes.c_int),
        ("amount", ctypes.c_float),
    ]


class Signature(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("Features", ctypes.POINTER(Feature)),
        ("Weights", ctypes.POINTER(ctypes.c_float)),
    ]


lib_path = os.path.join(os.path.dirname(__file__), "emd_c/libemd.so")
lib_emd = ctypes.CDLL(lib_path)
lib_emd.emd.argtypes = [
    ctypes.POINTER(Signature),
    ctypes.POINTER(Signature),
    ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(Feature), ctypes.POINTER(Feature)),
    ctypes.POINTER(Flow),
    ctypes.POINTER(ctypes.c_int),
]
lib_emd.emd.restype = ctypes.c_float


def dist_func(feature1, feature2):
    np_feature1 = np.ctypeslib.as_array(feature1.contents.centroid, shape=(24,))
    np_feature2 = np.ctypeslib.as_array(feature2.contents.centroid, shape=(24,))

    return np.linalg.norm(np_feature1 - np_feature2, ord=1)


DIST_FUNC_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_float, ctypes.POINTER(Feature), ctypes.POINTER(Feature)
)
dist_func_c = DIST_FUNC_TYPE(dist_func)


def create_signature_from_numpy(n, features_array, weights_array):
    signature = Signature()
    signature.n = ctypes.c_int(n)
    signature.Features = (Feature * n)()

    for i in range(n):
        feature = Feature()
        feature.centroid = (ctypes.c_float * 24)(*features_array[i])
        signature.Features[i] = feature

    signature.Weights = (ctypes.c_float * n)(*weights_array)
    return signature


class EMD:
    def __init__(
        self, frequencies=[0.1, 0.2, 0.3, 0.4], num_orientations=6, patch_size=128
    ):
        """
        Args:
            frequencies (python list): List of frequencies to be used 
                                    for Gabor's filter dictionary.
                                    The frequencies affect the scale of
                                    the extracted features. 
                                    By default we use [0.1, 0.2, 0.3, 0.4].
            num_orientations (int): Number of orientations to be used for
                                    the Gabor filter dictionary, 
                                    degrees are calculated with a constant step
                                    between 0 and pi. 
                                    Orientations affect the direction
                                    of the textures to be extracted.
                                    By default we use 6 orientations.
            patch_size (int): Size of windows on which to apply Gabor.
                              Default: 128x128. Change this parameter
                              if your images are smaller than 128x128
                              or if they are very large.
        """
        self.frequencies = frequencies
        self.orientations = np.linspace(0, np.pi, num_orientations)
        self.patch_size = patch_size
        self.gray_img1 = None
        self.signature_img1 = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_filters = len(frequencies) * len(self.orientations)
        self.magn_dict = {}
        self.l_pad = 0
        self.r_pad = 0
        self.t_pad = 0
        self.b_pad = 0
        self.mgn_diffs_save = None

        if frequencies != [0.1, 0.2, 0.3, 0.4] and num_orientations!=6:
            self.gabox_max_size = 0
            for frequency in self.frequencies:
                for theta in self.orientations:
                    kernel_test = gabor_kernel(frequency=frequency, theta=theta, bandwidth=1.0)
                    self.gabox_max_size = max(self.gabox_max_size, kernel_test.shape[0])
        else:
            self.gabox_max_size = 35
        self.gabor_weights = self._build_gabor_bank()

    def _build_gabor_bank(self):
        all_real_kernels = []
        all_imag_kernels = []
        for frequency in self.frequencies:
            for theta in self.orientations:
                kernel_complex = gabor_kernel(frequency=frequency, theta=theta, bandwidth=1.0)
                current_size = kernel_complex.shape[0]
                if current_size < self.gabox_max_size:
                    pad_size = self.gabox_max_size - current_size  # Es. 35 - 9 = 26
                    pad_top = pad_size // 2            # 26 // 2 = 13
                    pad_bottom = pad_size - pad_top    # 26 - 13 = 13
                
                    def pad_kernel(kernel_part):
                        return np.pad(
                            kernel_part, 
                            ((pad_top, pad_bottom), (pad_top, pad_bottom)), 
                            mode='constant', 
                            constant_values=0
                        )
                
                    kernel_real_np = pad_kernel(np.real(kernel_complex))
                    kernel_imag_np = pad_kernel(np.imag(kernel_complex))
                
                else:
                    kernel_real_np = np.real(kernel_complex)
                    kernel_imag_np = np.imag(kernel_complex)

                kernel_real = torch.from_numpy(kernel_real_np).float().to(self.device).view(1, 1, *kernel_real_np.shape)
                kernel_imag = torch.from_numpy(kernel_imag_np).float().to(self.device).view(1, 1, *kernel_imag_np.shape)
                
                all_real_kernels.append(kernel_real)
                all_imag_kernels.append(kernel_imag)

        gabor_weights = torch.cat(all_real_kernels + all_imag_kernels, dim=0)
        return gabor_weights

    def set_img1(self, img1):
        """
        Function that extracts texture signature on a single image. It is useful if img1 must always be the same.

        Args:
            img1 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns, to convert in Grayscale for signature extraction.
                            Required an image with values in the range [0, 255].
        """
        if not isinstance(img1, np.ndarray):
            raise TypeError(
                "Expected numpy.ndarray for input, but got " + str(type(img1))
            )

        if len(img1.shape) != 3:
            raise RuntimeError(
                "Expected 3D ndarray for input, but got a "
                + str(len(img1.shape))
                + "D ndarray"
            )

        if img1.shape[-1] != 3:
            raise RuntimeError(
                "Expected 3D ndarray for input with a shape of ("
                + str(img1.shape[0])
                + ", "
                + str(img1.shape[1])
                + ", 3), but got an ndarray with a shape of "
                + str(img1.shape)
            )

        if not np.issubdtype(img1.dtype, np.uint8):
            raise ValueError(
                "Expected 3D ndarray with np.unint8 values, but got an ndarray with a dtype "
                + str(img1.dtype)
            )

        self.gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gabor_response_img1 = self.apply_gabor_to_patches(self.gray_img1, 'img1')
        signature = self.calculate_image_signature(gabor_response_img1)
        self.signature_img1 = create_signature_from_numpy(
            signature[0].shape[0], signature[0], signature[1]
        )

    def extract_gabor_energy_batch(self, patches_list, id):
        """
        Estrae la matrice di energia normalizzata per tutte le patch in un'unica operazione batch.
        
        args:
            patches_list: array NumPy (H, W) of <N_patches> grayscale patches.
        return:
            Array NumPy with shape (N_patches, Num_Filters).
        """
        
        if not patches_list:
            return np.array([])

        # NumPy array to Tensor (N_patches, H, W)
        patches_np = np.stack(patches_list)
        #self.plot_patch_overview(patches_np, id)
        # Tensor (N, C, H, W) -> (N_patches, 1, H, W)
        input_tensor = torch.from_numpy(patches_np).float().to(self.device).unsqueeze(1)

        # Parallel convolution
        # response_combined shape (N_patches, Num_Filters * 2, H, W)
        response_combined = F.conv2d(input_tensor, self.gabor_weights, padding='same') #before padding was 'same'
        real_responses = response_combined[:, :self.num_filters, :, :]
        imag_responses = response_combined[:, self.num_filters:, :, :]

        pad = self.gabox_max_size // 2 
        real_responses = real_responses[:, :, pad:-pad, pad:-pad]
        imag_responses = imag_responses[:, :, pad:-pad, pad:-pad]

        #print('len rensponses: ', real_responses.shape)
        # Non-Normalized Energy
        # Tensor shape (N_patches, Num_Filters, H, W)
        magnitude_squared = real_responses**2 + imag_responses**2
        # print('avg value: ', avg_value)
        magnitude_squared_scaled = torch.log10(magnitude_squared + 1e-12)
        
        magn_recombined = []
        for i in range(self.num_filters):
            magn_recombined.append( self.plot_patch_overview(magnitude_squared_scaled[:, i, :, :], "filter_"+id+'_'+str(i), pad = 0) )
        self.magn_dict[id] = magn_recombined

        feature_matrix_vector = magnitude_squared.sum(dim=[2, 3]) 
        # acc_vector shape (N_patches, 1)
        acc_vector = feature_matrix_vector.sum(dim=1, keepdim=True) + sys.float_info.min
        # Normalizzations for each row (patch)
        # Tensor (N_patches, Num_Filters)
        energie_matrix_vector = feature_matrix_vector / acc_vector
        is_nan = torch.isnan(energie_matrix_vector)
        energie_matrix_vector = torch.where(is_nan, torch.tensor(0.0, device=self.device), energie_matrix_vector)

        feature_array_np = energie_matrix_vector.cpu().numpy().astype(np.float128)

        return feature_array_np


    def apply_gabor_to_patches(self, image, id):
        """
        Function that divides the image into patches and applies 
        Gabor's filter dictionary to each patch.

        Args:
            image (np.ndarray): A Grayscale image with shape [N, M] with N rows and M columns.
                            Required an image with values in the range [0, 255].

        Returns:
            Feature Matrix List (np.array): A list of Gabor response matrices for each patch.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Expected numpy.ndarray for input, but got " + str(type(image))
            )

        if len(image.shape) != 2:
            raise RuntimeError(
                "Expected 2D ndarray for input, but got a "
                + str(len(image.shape))
                + "D ndarray"
            )

        if not np.issubdtype(image.dtype, np.uint8):
            raise ValueError(
                "Expected 2D ndarray with np.unint8 values, but got an ndarray with a dtype "
                + str(image.dtype)
            )

        patches = self.divide_into_patches(image)
        return self.extract_gabor_energy_batch(patches, id)

    def divide_into_patches(self, image):
        """
        Function that divides the image into patches.

        Args:
            image (np.ndarray): A Grayscale image with shape [N, M] with N rows and M columns.
                            Required an image with values in the range [0, 255].

        Returns:
            Patches List (np.array): A list of patches of size: patch_size x patch_size.
        """
        patches = []
        img_h, img_w = image.shape
        rest_w = img_w%self.patch_size
        rest_h = img_h%self.patch_size

        if rest_w > 0:
            total_pad_w = self.patch_size - rest_w
            self.l_pad = math.ceil(total_pad_w / 2)
            self.r_pad = total_pad_w - self.l_pad
            image = cv2.copyMakeBorder(image,0,0, self.l_pad, self.r_pad,cv2.BORDER_CONSTANT,value=[0,0,0])
        if rest_h > 0:
            total_pad_h = self.patch_size - rest_h
            self.t_pad = math.ceil(total_pad_h / 2)
            self.b_pad = total_pad_h - self.t_pad
            image = cv2.copyMakeBorder(image, self.t_pad, self.b_pad,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])

        img_h, img_w = image.shape
        self.num_patch_on_col = int(img_h/self.patch_size)
        self.num_patch_on_row = int(img_w/self.patch_size)
        for i in range(0, img_h, self.patch_size):
            for j in range(0, img_w, self.patch_size):
                patch = image[i : i + self.patch_size, j : j + self.patch_size]
                if patch.shape == (self.patch_size, self.patch_size):
                    pad = self.gabox_max_size // 2
                    patch_padded = np.pad(patch, pad, mode='reflect')
                    patches.append(patch_padded)   

        return patches

    def combined_avg_magnitudes(self):
        magn_img1_list = self.magn_dict['img1'] #magnitude lists of img1
        magn_img2_list = self.magn_dict['img2'] #magnitude lists of img2

        diffs_list = []
        for i in range(self.num_filters):
            mag1_norm = (magn_img1_list[i] - magn_img1_list[i].min()) / (magn_img1_list[i].max() - magn_img1_list[i].min() + 1e-8)
            mag2_norm = (magn_img2_list[i] - magn_img2_list[i].min()) / (magn_img2_list[i].max() - magn_img2_list[i].min() + 1e-8)
            diffs_list.append( abs(mag1_norm - mag2_norm) )

        diffs = np.stack(diffs_list, axis=0)
        heatmap = np.mean(diffs, axis=0)

        if heatmap.max() == 0:
            self.mgn_diffs_save = heatmap.astype(np.uint8)
        else:
            self.mgn_diffs_save = (heatmap / heatmap.max() * 255).astype(np.uint8)


    def plot_patch_overview(self, patches, name, pad = 4):
        #patch size (Num_patches, H, W)
        tot_h = self.patch_size * self.num_patch_on_col
        tot_w = self.patch_size * self.num_patch_on_row
        if torch.is_tensor(patches):
            np_patches = patches.detach().cpu().numpy()
        else:
            np_patches = patches
        tot_h = np_patches.shape[1]*self.num_patch_on_col
        tot_w = np_patches.shape[2]*self.num_patch_on_row

        pad_rows = (self.num_patch_on_col+1)*pad
        pad_cols = (self.num_patch_on_row+1)*pad
        path_overview = np.full((tot_h+pad_rows-self.t_pad-self.b_pad, tot_w+pad_cols-self.l_pad-self.r_pad, 3), (0, 0, 255), dtype=np.uint8)
        o_i = pad
        o_j = pad
        idp = 0
        row_ind = 0
        for i in range(0, self.num_patch_on_col):
            o_j = pad
            col_ind = 0
            for j in range(0, self.num_patch_on_row):
                tp, bp, lp, rp = 0, 0, 0, 0
                if i == 0:
                    tp = self.t_pad
                elif i == self.num_patch_on_col-1:
                    bp = self.b_pad
                if j == 0:
                    lp = self.l_pad
                elif j == self.num_patch_on_row-1:
                    rp = self.r_pad

                patch = np_patches[idp, tp:(-bp or None), lp:(-rp or None)]
                idp +=1
                
                path_overview[row_ind+o_i : row_ind+o_i + patch.shape[0], col_ind+o_j : col_ind+o_j + patch.shape[1], 0] = patch.copy()
                path_overview[row_ind+o_i : row_ind+o_i + patch.shape[0], col_ind+o_j : col_ind+o_j + patch.shape[1], 1] = patch.copy()
                path_overview[row_ind+o_i : row_ind+o_i + patch.shape[0], col_ind+o_j : col_ind+o_j + patch.shape[1], 2] = patch.copy()

                o_j += pad
                col_ind += patch.shape[1]
            o_i += pad
            row_ind += patch.shape[0]
        cv2.imwrite("overview_test_"+name+".png", path_overview)
        return path_overview[:, :, 0]

    def calculate_image_signature(self, feature_vectors):
        """
        Function that compute MengHee-Heng Clustering on the list of 
        Gabor response matrices for each patch

        Args:
            feature_vectors (np.array): A list of Gabor response matrices 
                                        for each patch.

        Returns:
            centroids (np.array): A list of k centroids.
            weights (np.array): A list of corrispective k weights.
        """
        centroids, cluster = MengHeeHeng().fit(feature_vectors)
        weights = [len(l) for l in cluster]
        return np.array(centroids), np.array(weights)

    def compute_one_img(self, img2):
        """
        Function that extracts texture signature on img2 and compute emd distance 
        between signature_img1 and signature_img2. It is only useful if img1 is 
        always the same while img2 can change.

        Args:
            img2 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and 
                            M columns, to convert in Grayscale for signature
                            extraction. 
                            Required an image with values in the range [0, 255].
        Returns:
            score (float): The emd score between textures of img1 and img2.

        Example:
            emd = EMD(frequencies=[0.1, 0.2, 0.3, 0.4], num_orientations=6, patch_size=128)
            emd.set_img1(im1_bgr)
            score = emd.compute_one_img(img2_bgr)
        """
        if not isinstance(img2, np.ndarray):
            raise TypeError(
                "Expected numpy.ndarray for input, but got " + str(type(img2))
            )

        if len(img2.shape) != 3:
            raise RuntimeError(
                "Expected 3D ndarray for input, but got a "
                + str(len(img2.shape))
                + "D ndarray"
            )

        if img2.shape[-1] != 3:
            raise RuntimeError(
                "Expected 3D ndarray for input with a shape of ("
                + str(img2.shape[0])
                + ", "
                + str(img2.shape[1])
                + ", 3), but got an ndarray with a shape of "
                + str(img2.shape)
            )

        if not np.issubdtype(img2.dtype, np.uint8):
            raise ValueError(
                "Expected 3D ndarray with np.unint8 values, but got an ndarray with a dtype"
                + str(img2.dtype)
            )

        if self.signature_img1 is None:
            raise ValueError(
                "You did not set image 1. Run the command: help(compute_one_img)"
                +" to check how to run the code correctly."
            )

        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gabor_gpu = self.apply_gabor_to_patches(gray_img2, 'img2')
        self.combined_avg_magnitudes()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        signature = self.calculate_image_signature(gabor_gpu)
        signature_img2 = create_signature_from_numpy(
            signature[0].shape[0], signature[0], signature[1]
        )
        score = lib_emd.emd(
            ctypes.byref(self.signature_img1),
            ctypes.byref(signature_img2),
            dist_func_c,
            None,
            ctypes.c_int(0),
        )
        return score

    def compute_two_img(self, img1, img2):
        """
        Function that extracts texture signature on img1 and img2 and 
        compute emd distance between signature_img1 and signature_img2. 
        It is useful if img1 is not always the same and changes as img2.

        Args:
            img1 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns,
                            to convert in Grayscale for signature extraction.
                            Required an image with values in the range [0, 255].
            img2 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns,
                            to convert in Grayscale for signature extraction.
                            Required an image with values in the range [0, 255].
        Returns:
            score (float): The emd score between textures of img1 and im2.
        """
        self.set_img1(img1)
        return self.compute_one_img(img2)