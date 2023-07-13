# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tool for creating ZIP/PNG based datasets."""

from audioop import avg
from cProfile import label
import functools
import gzip
import io
import itertools
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from typing import Callable, Optional, Tuple, Union


import click
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from PIL import Image
from tqdm import tqdm

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    
    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname)) #, filename=fname)
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python # pylint: disable=import-error
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):
            return open_lmdb(source, max_images=max_images)
        else:
            return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if os.path.basename(source) == 'cifar-10-python.tar.gz':
            return open_cifar10(source, max_images=max_images)
        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------


###-------------- CALCULATION OF DENSITY IN CROWD IMAGES -----------------###
# The following functions are included for the calculation of density in
# crowd images labelled with bounding boxes (bbs), a requirement of the MSc
# research project 'Generative Models for the Synthesis and Manipulation of
# Crowded Scenes' by Philip Lord for the University of Lincoln.

def get_bbs(annotation):
    ''' Return a list of bbs from a CrowdHuman dataset annotations.odgt file. '''

    bbs = []

    # Extract bounding boxes from annotation file line
    ann_json = json.loads(annotation)

    for gtbox in ann_json['gtboxes']:
        # Get full bb [x, y, w, h] (includes non-visible parts of people)
        xywh = gtbox['fbox']
        
        # Calculate bb [x, y, x, y]
        xyxy = xywh
        xyxy[2] = xywh[0] + xywh[2] # botton right x
        xyxy[3] = xywh[1] + xywh[3] # bottom right y
        
        bbs.append(xyxy)

    return bbs


def get_centroids(bbs):
    ''' Return a list of centroids from a list of bbs (xyxy format). '''

    centroids = []

    for bb in bbs:
        centroids.append(((int(bb[2])+int(bb[0]))//2,(int(bb[3])+int(bb[1]))//2))

    return centroids


def calculate_avg_bb_size(bbs, img_w, img_h):
    ''' Return the average bb area relative to image size and in pixels. '''

    # Sum the area of each bb
    total_bb_area = 0
    for bb in bbs:
        x_min, y_min, x_max, y_max = bb
        bb_width = x_max - x_min
        bb_height = y_max - y_min
        total_bb_area += bb_width * bb_height

    # Calculate average bb area
    total_image_area = img_w * img_h
    avg_bb_area = total_bb_area / len(bbs)
    avg_bb_size = np.sqrt(avg_bb_area / total_image_area)

    return avg_bb_size, avg_bb_area


def calculate_avg_bb_dimensions(bbs):
    ''' Return the average bb width and height in pixels. '''

    total_width = 0
    total_height = 0

    # Sum the height of each bb
    for bb in bbs:
        x_min, y_min, x_max, y_max = bb

        bb_width = x_max - x_min
        total_width += bb_width

        bb_height = y_max - y_min
        total_height += bb_height

    # Calculate average bb width and height
    avg_bb_width = total_width / len(bbs)
    avg_bb_height = total_height / len(bbs)

    return avg_bb_width, avg_bb_height


def threshold_density(person_count):
    ''' Return the density label using manual thresholding based on person count. '''

    # Establish label based on sparse, medium, and dense thresholds
    if person_count <= 10:
        label = 0
    elif person_count > 10 and person_count <= 30:
        label = 1
    else:
        label = 2

    return label


def normalised_density(count, person_counts):
    ''' Return the density label by normalising person counts and thresholding in thirds. '''

    # Calculate density as a normalised value 
    density = (count - min(person_counts)) / (max(person_counts) - min(person_counts))

    # Thresholding
    if density < 0.333:
        label = 0
    elif density > 0.666:
        label = 2
    else:
        label = 1

    return label


def euclidean_density(bbs):
    ''' Return the density of bbs in a crowd image using euclidean distance. '''

    # https://github.com/ChargedMonk/Social-Distancing-using-YOLOv5/blob/b4694503d490797592f50a52aa7a7f7448c6bf58/utils/utils.py#L929
    # Calculate distances between every combination of 2 people
    bb_combos = list(itertools.combinations(bbs,2))
    distances = []

    for combo in bb_combos:
        # Calculate centroids
        bb1, bb2 = combo[0], combo[1]
        centroid1 = ((int(bb1[2])+int(bb1[0]))//2,(int(bb1[3])+int(bb1[1]))//2)
        centroid2 = ((int(bb2[2])+int(bb2[0]))//2,(int(bb2[3])+int(bb2[1]))//2)

        # Calculate distance between centroids
        distance = ((centroid2[0]-centroid1[0])**2 + (centroid2[1]-centroid1[1])**2)**0.5
        distances.append(distance)

    # Calculate average distance
    avg_distance = np.mean(distances)

    # Calculate density using average bb width
    avg_bb_width, _ = calculate_avg_bb_dimensions(bbs)
    density = avg_distance / avg_bb_width

    # Assign density label based on threshold values
    if density < 1.5:
        label = 0  # Low Density
    elif density >= 3:
        label = 2  # High Density
    else:
        label = 1  # Medium Density

    print(f"Density: {density}\nLabel: {label}\n")

    return label


def grid_density(bbs, img_w, img_h):
    ''' Return the density of bbs in a crowd image using adaptive cell size grid clustering. '''

    # Calculate average bb size
    avg_bb_size, _ = calculate_avg_bb_size(bbs, img_w, img_h)
    
    # Determine desired cell size based on average bb size (adaptive cell size)
    desired_num_cells = 25
    cell_x = img_w / desired_num_cells
    cell_y = img_h / desired_num_cells
    cell_size = max(cell_x, cell_y, avg_bb_size * 168.5)

    # Calculate no. cells based on image size
    num_cells_x = int(img_w / cell_size)
    num_cells_y = int(img_h / cell_size)
    
    # Create grid using cell size and no.
    grid = np.zeros((num_cells_y, num_cells_x))

    # Count no. bbs in each cell
    for bb in bbs:
        x_min, y_min, x_max, y_max = bb
        cell_x_min = int(x_min / cell_size)
        cell_y_min = int(y_min / cell_size)
        cell_x_max = int(x_max / cell_size)
        cell_y_max = int(y_max / cell_size)

        # Check if cell indices are within grid dimensions
        cell_x_min = max(cell_x_min, 0)
        cell_y_min = max(cell_y_min, 0)
        cell_x_max = min(cell_x_max, num_cells_x - 1)
        cell_y_max = min(cell_y_max, num_cells_y - 1)

        # Increment cell count for bounding box presence
        grid[cell_y_min:cell_y_max + 1, cell_x_min:cell_x_max + 1] += 1

    # Perform clustering on grid using connected components
    crowded_regions, num_clusters = ndimage.label(grid > 0)

    # Calculate density for each cluster
    # Cluster density = avg no. people per cell within cluster (no. people in cluster / no. cells in cluster)
    cluster_densities = []
    for cluster_label in range(1, num_clusters + 1):
        cluster_mask = crowded_regions == cluster_label
        cluster_densities.append(np.sum(grid[cluster_mask]) / np.sum(cluster_mask) if np.sum(cluster_mask) > 0 else 0)

    # Print individual cluster densities
    for cluster_label, cluster_density in enumerate(cluster_densities):
        print(f"Cluster {cluster_label + 1} Density: {cluster_density}")
    
    # Find densest cluster
    density = np.max(cluster_densities)

    # Assign density label based on threshold values
    if density < 2:
        label = 0  # Low Density
    elif density >= 3:
        label = 2  # High Density
    else:
        label = 1  # Medium Density

    print(f"Density: {density}\nLabel: {label}\n")

    return label


def grid_density_metres(bbs, img_w, img_h):
    ''' Return the density of bbs in a crowd image using 1m^2 cell grid clustering. '''
        
    # Get average person height
    _, avg_bb_height = calculate_avg_bb_dimensions(bbs)

    # Calculate 1m with respect to image using average height of 168.5cm
    metre = (avg_bb_height / 168.5) * 100
    
    # Make cell size = 1m^2
    num_cells_x = int(img_w / metre)
    num_cells_y = int(img_h / metre)

    # Create grid using cell size and no.
    grid = np.zeros((num_cells_y, num_cells_x))

    # Count no. bbs in each cell
    for bb in bbs:
        x_min, y_min, x_max, y_max = bb
        cell_x_min = int(x_min / metre)
        cell_y_min = int(y_min / metre)
        cell_x_max = int(x_max / metre)
        cell_y_max = int(y_max / metre)

        # Check if cell indices are within grid dimensions
        cell_x_min = max(cell_x_min, 0)
        cell_y_min = max(cell_y_min, 0)
        cell_x_max = min(cell_x_max, num_cells_x - 1)
        cell_y_max = min(cell_y_max, num_cells_y - 1)

        # Increment cell count for bounding box presence
        grid[cell_y_min:cell_y_max+1, cell_x_min:cell_x_max+1] += 1

    # Perform clustering on grid using connected components
    crowded_regions, num_clusters = ndimage.label(grid > 0)

    # Calculate density for each cluster
    # Cluster density = avg no. people per cell within cluster
    cluster_densities = []
    for cluster_label in range(1, num_clusters + 1):
        cluster_mask = crowded_regions == cluster_label
        cluster_density = np.sum(grid[cluster_mask]) / np.sum(cluster_mask)
        cluster_densities.append(cluster_density)

    # Print individual cluster densities
    for cluster_label, cluster_density in enumerate(cluster_densities):
        print(f"Cluster {cluster_label + 1} Density: {cluster_density}")

    # Find densest cluster
    density = np.max(cluster_densities)

    # Assign density label based on threshold values
    if density < 2:
        label = 0  # Low Density
    elif density >= 3:
        label = 2  # High Density
    else:
        label = 1  # Medium Density

    print(f"Density: {density}\nLabel: {label}\n")

    return label


def kde_density(bbs, img_w, img_h):
    ''' Return the density of bbs in a crowd image using KDE clustering. '''

    # Set bandwidth and step size for kernel
    bandwidth = 0.5
    step_size = 1

    # Convert centroids to np array
    centroids = np.array(get_centroids(bbs))

    # Create KDE instance
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')

    # Fit KDE model to data
    kde.fit(centroids)

    # Generate grid of points covering entire image to evaluate density
    x, y = np.meshgrid(np.arange(0, img_w, step_size),
                       np.arange(0, img_h, step_size))
    xy = np.vstack([x.ravel(), y.ravel()]).T

    # Calculate density values for the grid points
    density_values = np.exp(kde.score_samples(xy))

    # Reshape density values to match shape of grid
    density_map = density_values.reshape(x.shape)

    plt.figure(figsize=(8, 6))
    plt.imshow(density_map, cmap='hot', origin='lower')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Density Map')
    plt.show()

    # Perform clustering on grid using connected components
    crowded_regions, num_clusters = ndimage.label(density_map > 0)

    # Calculate density for each cluster
    # Cluster density = avg no. people per cell within cluster (no. people in cluster / no. cells in cluster)
    cluster_densities = []
    for cluster_label in range(1, num_clusters + 1):
        cluster_mask = crowded_regions == cluster_label
        cluster_density = np.sum(density_map[cluster_mask]) / np.sum(cluster_mask)
        cluster_densities.append(cluster_density)

    # Print individual cluster densities
    for cluster_label, cluster_density in enumerate(cluster_densities):
        print(f"Cluster {cluster_label + 1} Density: {cluster_density}")
    
    # Find densest cluster
    density = np.max(cluster_densities)

    # Assign density label based on threshold values
    if density < 2:
        label = 0  # Low Density
    elif density >= 3:
        label = 2  # High Density
    else:
        label = 1  # Medium Density

    print(f"Density: {density}\nLabel: {label}\n")

    return label


def generate_ch_labels(source, method):
    ''' Create a dataset.json file in source with crowd density labels using CrowdHuman annotations.odgt. '''

    # Read CrowdHuman annotations
    with open(os.path.join(source, 'annotations.odgt'), 'r') as file:
        annotations = file.readlines()
    
    # Declare labels['labels'] dictionary for dataset.json file
    labels = {}
    labels['labels'] = []

    # Get person count of each image for normalised method
    if method == 'normalised':
        person_counts = [annotation.count('person') for annotation in annotations]
    
    # Calculate density of each image and generate label
    for annotation in annotations:
        # Get image name, height, and width
        img_name = json.loads(annotation)['ID'] + '.jpg'
        img = Image.open(os.path.join(source, img_name))
        img_w, img_h = img.size

        # Extract bbs from image annotation
        bbs = get_bbs(annotation)

        # Calculate density using selected method
        if method == 'threshold':
            density_label = threshold_density(len(bbs))
        elif method == 'normalised':
            density_label = normalised_density(len(bbs), person_counts)
        elif method == 'euclidean':
            density_label = euclidean_density(bbs)
        elif method == 'grid':
            density_label = grid_density(bbs, img_w, img_h)
        elif method == 'grid-metres':
            density_label = grid_density_metres(bbs, img_w, img_h)
        elif method == 'kde':
            density_label = kde_density(bbs, img_w, img_h)

        # Append label to dictionary
        labels['labels'].append([img_name, density_label])
    
    # Create dataset.json file in source using label dictionary
    with open(os.path.join(source, 'dataset.json'), 'w') as file:
        json.dump(labels, file)

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
@click.option('--density', help='Prepare dataset.json for CrowdHuman dataset', type=click.Choice(['threshold', 'normalised', 'euclidean', 'grid', 'grid-metres', 'kde']))
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    density: Optional[str]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    To prepare the 'dataset.json' file with crowd densities calculated from a CrowdHuman
    'annotations.odgt' file, use the --density option and specify the method used to
    calculate crowd density.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --resolution=512x384
    """

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    # Create dataset.json labels file if dataset is CrowdHuman
    if density is not None:
        generate_ch_labels(source, density)
        
    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    if resolution is None: resolution = (None, None)
    transform_image = make_transform(transform, *resolution)

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_image(image['img'])

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()] # pylint: disable=unsubscriptable-object
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
