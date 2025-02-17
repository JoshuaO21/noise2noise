# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# See README.md in this directory for instructions on how to use this script.

import re
import argparse
import glob
import os
import PIL.Image
import numpy as np
import sys

import util

import nibabel as nib

OUT_RESOLUTION = 256

# Select z-slices from [25,124]
slice_min = 25
slice_max = 125

def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

def undersample_kspace(slice, mask_fraction=0.5):
    """ Apply undersampling by masking out a fraction of k-space. """
    fft_slice = np.fft.fft2(slice)
    fft_slice = np.fft.fftshift(fft_slice)
    mask = np.random.rand(*fft_slice.shape) < mask_fraction
    undersampled_fft = fft_slice * mask
    undersampled_fft = np.fft.ifftshift(undersampled_fft)
    return np.abs(np.fft.ifft2(undersampled_fft))

def genpng(args):
    if args.outdir is None:
        print ('Must specify output directory with --outdir')
        sys.exit(1)
    if args.ixi_dir is None:
        print ('Must specify input IXI-T1 directory with --ixi-dir')
        sys.exit(1)

    mri_directory = args.ixi_dir
    out_directory = args.outdir
    os.makedirs(out_directory, exist_ok=True)

    nii_files = glob.glob(os.path.join(mri_directory, "*.nii.gz"))

    for nii_file in nii_files:
        print('Processing', nii_file) 
        nii_img = nib.load(nii_file)
        name = os.path.basename(nii_file).split(".")[0]
        print("name", name)
        hborder = (np.asarray([OUT_RESOLUTION, OUT_RESOLUTION]) - nii_img.shape[0:2]) // 2
        print("Img: ", nii_img.shape, " border: ", hborder)
        
        img = nii_img.get_fdata().astype(np.float32)
        img = img / np.max(img)
        print('Max value', np.max(img))

        for s in range(slice_min, slice_max):
            slice = img[:, :, s]
            
            if args.undersample:
                slice = undersample_kspace(slice)
            
            output = np.zeros([OUT_RESOLUTION, OUT_RESOLUTION])
            output[hborder[0] : hborder[0] + nii_img.shape[0], hborder[1] : hborder[1] + nii_img.shape[1]] = slice
            output = np.minimum(output, 1.0)
            output = np.maximum(output, 0.0)
            output = output * 255

            if np.max(output) > 1.0:
                outname = os.path.join(out_directory, "%s_%03d.png" % (name, s))
                PIL.Image.fromarray(output).convert('L').save(outname)

def main():
    parser = argparse.ArgumentParser(
        description='Convert the IXI-T1 dataset into a format suitable for network training'
    )
    subparsers = parser.add_subparsers(help='Sub-commands')
    parser_genpng = subparsers.add_parser('genpng', help='IXI nifti to PNG converter (intermediate step)')
    parser_genpng.add_argument('--ixi-dir', help='Directory pointing to unpacked IXI-T1.tar')
    parser_genpng.add_argument('--outdir', help='Directory where to save .PNG files')
    parser_genpng.add_argument('--undersample', action='store_true', help='Apply k-space undersampling to slices')
    parser_genpng.set_defaults(func=genpng)

    args = parser.parse_args()
    if 'func' not in args:
        print ('No command given.  Try --help.')
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
