# ------------------------------------------------------------------------
# EDOK Metric
# Authors: Vincenzo Bevilacqua, Antonio Di Marino 
# Copyright (c) 2026. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
import numpy as np
import torch

class Oklab():
    def __init__(self):
        self.ok_img = None
        self.diff_map = None

    def set_img1(self, img):
        """
        Function that convert img1 to Oklab space. It is useful if img1 must always be the same.

        Args:
            img1 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns, 
                            to convert in Grayscale for signature extraction.
                            Required an image with values in the range [0, 255]. 
        """
        self.ok_img1 = self.bgr_to_oklab(img)
        
    def bgr_to_linear_srgb(self, img):
        """
        Convert an image from BGR-8bit to linear-sRGB as reported on https://www.w3.org/Translations/WCAG22-it/relative-luminance.html
        Compared with the above-mentioned site that traform the single color (pixel),
        the following function applies all steps to all pixels of the input image.

        Args:
            img (np.ndarray): BGR input image in the range [0-255] to be converted. 
                            img must have a shape [N, M ,3] with N rows and M columns.

        Returns:
            l_img (np.ndarray): linear-sRGB image in the range [0-1].
                                l_img is returned with the shape [M, N, 3]. 
        """

        if not isinstance(img, np.ndarray):
            raise TypeError('Expected numpy.ndarray for input, but got ' + str(type(img)))
        
        if len(img.shape) != 3:
            raise RuntimeError('Expected 3D ndarray for input, but got a ' + str(len(img.shape)) + 'D ndarray')
        
        if img.shape[-1] != 3:
            raise RuntimeError('Expected 3D ndarray for input with a shape of ('+str(img.shape[0])+', '+str(img.shape[1])+', 3), but got an ndarray with a shape of '+ str(img.shape))

        if not np.issubdtype(img.dtype, np.uint8):
            raise ValueError('Expected 3D ndarray with np.unint8 values, but got an ndarray with a dtype '+ str(img.dtype))
        
        #Torch operations are more accurate
        srgb_img = torch.from_numpy(img[:, :, ::-1].astype("float64")) / 255.0
        l_img = torch.where(srgb_img > 0.04045, ((srgb_img+0.055)/1.055)**2.4, srgb_img / 12.92)
        return l_img.detach().numpy()
    
    def linear_srgb_to_oklab(self, l_img):
        """
        Converts linear-sRGB to Oklab  as reported on https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab

        Args:
            l_img (np.ndarray): linear-sRGB image in the range [0, 1] with the shape (H, W, 3),
                                where H is the number of rows and W is the number of cols.
        
        Returns:
            ok_img (np.ndarray): Image in perceptual space Oklab . The output image has a shape of (H, W, 3),
                                where the first channel corresponds to the 'L' in the range [0, 1],
                                the second channel is the 'a' component generally in the range [-0.233, 0.276], 
                                and the third channel corresponds to the 'b' component generally in the range [-0.311, 0.198],
        """
    
        l = 0.4122214708 * l_img[:, :, 0] + 0.5363325363 * l_img[:, :, 1] + 0.0514459929 * l_img[:, :, 2]
        m = 0.2119034982 * l_img[:, :, 0] + 0.6806995451 * l_img[:, :, 1] + 0.1073969566 * l_img[:, :, 2]
        s = 0.0883024619 * l_img[:, :, 0] + 0.2817188376 * l_img[:, :, 1] + 0.6299787005 * l_img[:, :, 2]

        l_ = np.cbrt(l)
        m_ = np.cbrt(m)
        s_ = np.cbrt(s)

        ok_img = np.empty(l_img.shape, dtype=np.float64)
        ok_img[:, :, 0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        ok_img[:, :, 1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        ok_img[:, :, 2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        return ok_img

    def bgr_to_oklab(self, bgr_img):
        """
        Converts an image from BGR format to Oklab  color space, generally BGR is the format of images opened with the OpenCV library.
        As reported by Björn Ottosson on https://bottosson.github.io/posts/oklab/#the-oklab-color-space,
        colors in Oklab  space are represented by three components:
        L - perceived brightness
        a - how green/red the color is along the green-to-red axis of the four unique hues.
        b - how blue/yellow the color is along the blue-to-yellow axis of the four unique hues.

        Args:
            bgr_img (np.ndarray): BGR image (H x W x 3) in the range [0, 255],
                                where H is the number of rows and W is the number of cols.

        Returns:
            ok_img (np.ndarray): Image in the Oklab  color space (H x W x 3).
                                Where the first channel corresponds to the 'L' ranging from 0 (black) to 1 (white).
                                While the second channel 'a' and the third channel 'b' in Oklab are theoretically unbounded, 
                                but converting from a discrete and limited range of BGR colors,
                                then channel 'a' has values ranging from -0.23388757 (green) to 0.2762164 (bright magenta) 
                                and channel 'b' has values ranging from -0.31152815 (blue) to 0.19856975 (yellow).
        """

        l_img = self.bgr_to_linear_srgb(bgr_img)
        ok_img = self.linear_srgb_to_oklab(l_img)
        return ok_img
    
    def compute_one_img(self, img2):
        """
        Function that convert img2 to Oklab space and compute color distance between img1 and img2. 
        It is only useful if img1 is always the same while img2 can change.

        Args:
            img2 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns, 
                            to convert in Grayscale for signature extraction.
                            Required an image with values in the range [0, 255]. 
        Returns:
            score (float): The OK score between color of img1 and im2.
        """
        return self.oklab_dissimilarity(img2)

    def compute_two_img(self, img1, img2):
        """
        Function that convert img1 and img2 to Oklab space and compute color distance between img1 and img2. 
        It is useful if img1 is not always the same and changes as img2.

        Args:
            img1 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns, to convert in
                            Grayscale for signature extraction.
                            Required an image with values in the range [0, 255]. 
            img2 (np.ndarray): A BGR image with shape [N, M, 3] with N rows and M columns, to convert in
                            Grayscale for signature extraction.
                            Required an image with values in the range [0, 255]. 
        Returns:
            score (float): The OK score between color of img1 and im2.
        """
        self.set_img1(img1)
        return self.oklab_dissimilarity(img2)

    # Color-difference metric in OKLab space as reported on https://www.w3.org/TR/css-color-4/#color-difference-OK 
    # and https://github.com/svgeesus/svgeesus.github.io/blob/master/Color/OKLab-notes.md#color-difference-metric
    def oklab_dissimilarity(self, img2):
        """
        Color-difference metric in OKLab space as reported on
        https://www.w3.org/TR/css-color-4/#color-difference-OK

        Args:
            img2 (np.ndarray): Second BGR image with shape [N, M, 3] with N rows and M columns, 
                            to be compared in OKLab space.
                            Required an image with values in the range [0, 255]. 
        Returns:
            score (float): Score of dissimilarity in OKLab space averaged over the total number of pixels.
        Example:
            ok = OK()
            ok.set_img1(img1_bgr)
            score = ok.compute_one_image(img2_bgr)
        """
        if self.ok_img1 is None:
            print('You did not set image 1. Run the command: help(oklab_dissimilarity) to check how'
                 + 'to run the code correctly.')
        ok_img2 = self.bgr_to_oklab(img2)
        self.diff_map = np.linalg.norm(self.ok_img1 - ok_img2, axis=2)
        
        return np.mean(np.linalg.norm(self.ok_img1 - ok_img2, axis=2))