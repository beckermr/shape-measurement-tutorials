{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "13a6ff05",
   "metadata": {},
   "source": [
    "# Finding Good Background Pixels in DES Data\n",
    "\n",
    "We're gonna go through some of the basics on how to access good pixels which are also background pixels in DES data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6041f640",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import fitsio\n",
    "import yaml\n",
    "import esutil\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2ff3de5",
   "metadata": {},
   "source": [
    "## What is a good background pixel anyways?\n",
    "\n",
    "We will define the good background pixels as\n",
    "\n",
    "1. not associated with any detected objects\n",
    "2. positive weight\n",
    "3. no bit flags set\n",
    "\n",
    "Let's discuss these in more detail.\n",
    "\n",
    "### How can I tell if a pixel is associated with any detected objects?\n",
    "\n",
    "For this task, we will use the segmentation maps associated with the coadd and single-epoch images. Remember that \n",
    "a segementation image has integer values and marks areas of the image associated with each detection. For our purposes, we only want pixels where the segmentation map is 0.\n",
    "\n",
    "### What does it mean for a pixel to have positive weight?\n",
    "\n",
    "Remember that the weight map is the inverse of the variance of the data in the pixel. This format means that pixels with large variance have very small weights. Sometimes, we set the value of the weight map to zero by hand in order to indicate that a given pixel should be ignored. Thus we want to demand that the weight map is greater than zero.\n",
    "\n",
    "### What does it mean for a pixel to have no bit flags set?\n",
    "\n",
    "As images are processed, numerical operations are performed on the pixels and sets of pixels are flagged as being from artifacts, defects, etc. This information is typically stored in what is known as a bit mask image. A bit mask image is an image of integers. For each pixel, the underlying binary representation of the number is used to store flags indicating different meanings. If not bit flags are set, then value of this field should be 0.\n",
    "\n",
    "## How does this apply to DES data?\n",
    "\n",
    "For DES data, we will demand these conditions of both the single-epoch image and the coadd image in the same part of the sky. To do this we will have to map all of the single-epoch pixels to their nearest location in the coadd image. \n",
    "\n",
    "Below I've put some code in the help guide you through this task. The steps will be as follows.\n",
    "\n",
    "1. Read in the coadd image and single-epoch image data.\n",
    "2. Map all of the single-epoch pixels to their nearest coadd pixel.\n",
    "4. Make the proper set of cuts on all of these quantities. \n",
    "5. Visualize the resulting image.\n",
    "\n",
    "### 1. Read in the Image Data\n",
    "\n",
    "We'll use the wcs reading function from the last tutorial."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d0e4b53a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def read_wcs(pth, ext=0):\n",
    "    hdr = fitsio.read_header(pth, ext=ext)\n",
    "    dct = {}\n",
    "    for k in hdr.keys():\n",
    "        try:\n",
    "            dct[k.lower()] = hdr[k]\n",
    "        except Exception:\n",
    "            pass\n",
    "    return esutil.wcsutil.WCS(dct)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b9f1438b",
   "metadata": {},
   "source": [
    "Now let's get the images and the WCS solutions. First we grab the info dictionary from the YAML to get the paths."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f8902d96",
   "metadata": {},
   "outputs": [],
   "source": [
    "meds_dir = \"/cosmo/scratch/mrbecker/MEDS_DIR\"\n",
    "tilename = \"DES0124-3332\"\n",
    "band = \"i\"\n",
    "yaml_pth = os.path.join(\n",
    "    meds_dir, \n",
    "    \"des-pizza-slices-y6-v8/pizza_cutter_info/%s_%s_pizza_cutter_info.yaml\" % (\n",
    "        tilename, band\n",
    "    )\n",
    ")\n",
    "\n",
    "with open(yaml_pth, \"r\") as fp:\n",
    "    info = yaml.safe_load(fp.read())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ccd9dee4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "meds_dir = os.environ.get(\"MEDS_DIR\")\n",
    "tilename = \"DES2359-6331\"\n",
    "band = \"i\"\n",
    "yaml_pth = os.path.join(\n",
    "    meds_dir, \n",
    "    \"des-pizza-slices-y6-v6/pizza_cutter_info/%s_%s_pizza_cutter_info.yaml\" % (\n",
    "        tilename, band\n",
    "    )\n",
    ")\n",
    "\n",
    "with open(yaml_pth, \"r\") as fp:\n",
    "    info = yaml.safe_load(fp.read())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea794081",
   "metadata": {},
   "source": [
    "And we read the stuff:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "90c7c50c",
   "metadata": {},
   "outputs": [],
   "source": [
    "coadd_wcs = read_wcs(info['image_path'], ext=info['image_ext'])\n",
    "coadd_image = fitsio.read(info['image_path'], ext=info['image_ext'])\n",
    "coadd_weight = read_wcs(info['weight_path'], ext=info['weight_ext'])\n",
    "coadd_bmask = read_wcs(info['bmask_path'], ext=info['bmask_ext'])\n",
    "coadd_seg = read_wcs(info['seg_path'], ext=info['seg_ext'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50d5981c",
   "metadata": {},
   "source": [
    "We will look at the 5th single-epoch image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "5b8717e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "se_ind = 4  # 5th image is index 4\n",
    "si = info['src_info'][se_ind]\n",
    "se_wcs = read_wcs(si['image_path'], ext=si['image_ext'])\n",
    "se_image = fitsio.read(si['image_path'], ext=si['image_ext'])\n",
    "se_weight = fitsio.read(si['weight_path'], ext=si['weight_ext'])\n",
    "se_bmask = fitsio.read(si['bmask_path'], ext=si['bmask_ext'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f89602c",
   "metadata": {},
   "source": [
    "### 2. Map all of the single-epoch pixels to the nearest coadd pixel\n",
    "\n",
    "To get you started, I have build a list of the pixel indices for the single-epoch image pixels below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "26e543d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "xind, yind = np.meshgrid(se_image.shape[1], se_image.shape[0])\n",
    "xind = xind.ravel()\n",
    "yind = yind.ravel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12363efd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# do the rest here, computing the coadd indices for each single-epoch pixel\n",
    "# when you do this, you'll need to cut out indices less than zero or greater than or equal to the dimenions\n",
    "# this can be done with mask arrays\n",
    "#  msk = (x >= 0) & (x < coadd_image.shape[1])\n",
    "#  x = x[msk]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f099ceb4",
   "metadata": {},
   "source": [
    "### 3. Make the cuts on the proper quantities\n",
    "\n",
    "Once you have the pixel locations remapped, then you can make cuts using the same mask array syntax as above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62a0b6b3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "720165f9",
   "metadata": {},
   "source": [
    "### 4. Visualize the Data\n",
    "\n",
    "Make a plot of the good and bad pixels in the image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ece5c40f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:anl] *",
   "language": "python",
   "name": "conda-env-anl-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
