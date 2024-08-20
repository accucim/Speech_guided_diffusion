import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from mnist_normalize_meldataset import MnistMelDataset
import glob


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
):

    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)
    # print(all_files)
    classes = None
    
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        # print(all_files)
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        classes = class_names
        
    dataset = MnistMelDataset(
        training_files=all_files,
        segment_size=22050,
        sampling_rate=22050,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    # sampler = DistributedSampler(dataset, shuffle=not deterministic)
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader



def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["wav", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


# class ImageDataset(Dataset):
#     def __init__(
#         self,
#         image_size,
#         image_paths,
#         classes=None,
#         shard=0,
#         num_shards=1,
#     ):
#         super().__init__()
#         self.local_images = image_paths[shard:][::num_shards]
#         self.local_classes = None if classes is None else classes[shard:][::num_shards]

#     def __len__(self):
#         return len(self.local_images)

#     def __getitem__(self, idx):
#         path = self.local_images[idx]

#         out_dict = {}
#         if self.local_classes is not None:
#             out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            
#         return arr, out_dict

