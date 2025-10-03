import numpy as np
import matplotlib.pyplot as plt
import zarr
from matplotlib.patches import Circle
from skimage.draw import disk

def create_toy_dataset(location='test_dataset.zarr',shape=[512,512],num_frames=16,num_discs=4,disc_radius=20,zarr_chunks=(1, 512, 512, 5),moving_fraction=0.0,max_speed = 10):
    blank_image = np.zeros(shape + [num_frames], dtype=np.uint8)

    moving_discs = int(num_discs * moving_fraction)
    unmoving_discs = num_discs - moving_discs

    unmoving_coords = np.random.random((unmoving_discs,len(shape)))
    unmoving_coords *= shape
    unmoving_coords = unmoving_coords.astype(int)

    for disk_location in enumerate(unmoving_coords):
        rr, cc = disk(disk_location[1], disc_radius, shape=blank_image.shape[:2])
        blank_image[rr, cc,:] = 1

    for i in range(moving_discs):
        # disc_speed = (np.random.random(len(shape)) * (max_speed * 2) - max_speed).astype(int)
        disc_speed = [0, 1]
        initial_location = (np.random.random(len(shape)) * shape).astype(int)
        for frame in range(num_frames):
            rr, cc = disk(initial_location, disc_radius, shape=blank_image.shape[:2])
            blank_image[rr, cc, frame] = 1
            initial_location += disc_speed

    test_image = np.expand_dims(blank_image.transpose(2,0,1),1)

    root = zarr.open(location, mode='w')

    z = root.zeros('raw', shape=test_image.shape, chunks=zarr_chunks, dtype=test_image.dtype)
    z[:] = test_image
    
    
    print(f"Created dataset at {location} with shape {test_image.shape}")
    z.attrs["axis_names"] = ["s","c"] + ["z", "y", "x"][:len(shape)]
    z.attrs["resolution"] = (1,1,1,1)
    z.attrs["offset"] = (0,0,0,0)

def sphere_in_shape(radius, center, shape):
    """Create a sphere in a 3D array."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return np.nonzero((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2 <= radius**2)

def sphere(center, radius, shape = None):
    # center = np.array([x,y,z])
    radii_rot = np.array([radius, radius, radius])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:3]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    xx, yy, zz = sphere_in_shape(radius, shifted_center, bounding_shape)
    xx += upper_left[0]
    yy += upper_left[1]
    zz += upper_left[2]
    return xx, yy, zz


def create_3D_toy_dataset(location='test_dataset.zarr',shape=[512,512],num_frames=16,num_discs=4,disc_radius=20,zarr_chunks=(1, 512, 512, 5),moving_fraction=0.0,max_speed = 10):
    blank_image = np.zeros(shape + [num_frames], dtype=np.uint8)
    blank_GT = np.zeros(shape + [num_frames], dtype=np.uint8)

    moving_discs = int(num_discs * moving_fraction)
    unmoving_discs = num_discs - moving_discs

    unmoving_coords = np.random.random((unmoving_discs,len(shape)))
    unmoving_coords *= shape
    unmoving_coords = unmoving_coords.astype(int)

    GT_label = 1

    for disk_location in enumerate(unmoving_coords):
        if len(shape) == 2:
            rr, cc = disk(disk_location[1], disc_radius, shape=blank_image.shape[:2])
            blank_image[rr, cc,:] = 1
            blank_GT[rr, cc,:] = GT_label
        elif len(shape) == 3:
            xx, yy, zz = sphere(disk_location[1], disc_radius, shape=blank_image.shape[:3])
            blank_image[xx, yy, zz,:] = 1
            blank_GT[xx, yy, zz,:] = GT_label
        GT_label += 1

    for i in range(moving_discs):
        disc_speed = (np.random.random(len(shape)) * (max_speed * 2) - max_speed).astype(int)
        # disc_speed = [0, 1]
        initial_location = (np.random.random(len(shape)) * shape).astype(int)
        for frame in range(num_frames):
            if len(shape) == 2:
                rr, cc = disk(initial_location, disc_radius, shape=blank_image.shape[:2])
                blank_image[rr, cc, frame] = 1
                blank_GT[rr, cc, frame] = GT_label
            elif len(shape) == 3:
                xx, yy, zz = sphere(initial_location, disc_radius, shape=blank_image.shape[:3])
                blank_image[xx, yy, zz, frame] = 1
                blank_GT[xx, yy, zz, frame] = GT_label
            else:
                raise ValueError("Shape must be 2D or 3D.")
            # rr, cc = disk(initial_location, disc_radius, shape=blank_image.shape[:2])
            # blank_image[rr, cc, frame] = 1
            initial_location += disc_speed
        GT_label += 1

    # test_image = np.expand_dims(blank_image.transpose(2,0,1),1)
    test_image = np.expand_dims(blank_image.transpose((len(shape),)+(0,1,2)[:len(shape)]),1)
    test_GT = np.expand_dims(blank_GT.transpose((len(shape),)+(0,1,2)[:len(shape)]),1)

    root = zarr.open(location, mode='w')

    z = root.zeros('raw', shape=test_image.shape, chunks=zarr_chunks, dtype=test_image.dtype)
    z[:] = test_image

    z_GT = root.zeros('groundtruth', shape=test_GT.shape, chunks=zarr_chunks, dtype=test_GT.dtype)
    z_GT[:] = test_GT
    
    
    print(f"Created dataset at {location} with shape {test_image.shape}")
    z.attrs["axis_names"] = ["s","c"] + ["z", "y", "x"][:len(shape)]
    z.attrs["resolution"] = (1,1,1,1)
    # z.attrs["offset"] = (0,0,0,0)
    z_GT.attrs["axis_names"] = ["s","c"] + ["z", "y", "x"][:len(shape)]
    z_GT.attrs["resolution"] = (1,1,1,1)
    # z.attrs["offset"] = (0,0,0,0)