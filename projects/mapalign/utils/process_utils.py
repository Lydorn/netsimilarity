import sys

import skimage.io

sys.path.append("../../utils")
import polygon_utils


def rescale_data(image, polygons, scale):
    downsampled_image = skimage.transform.rescale(image, scale, order=3, preserve_range=True, multichannel=True,
                                                  anti_aliasing=True)
    downsampled_image = downsampled_image.astype(image.dtype)
    downsampled_polygons = polygon_utils.rescale_polygon(polygons, scale)
    return downsampled_image, downsampled_polygons


def downsample_data(image, metadata, polygons, factor, reference_pixel_size):
    corrected_factor = factor * reference_pixel_size / metadata["pixelsize"]
    scale = 1 / corrected_factor
    downsampled_image, downsampled_polygons = rescale_data(image, polygons, scale)
    return downsampled_image, downsampled_polygons
