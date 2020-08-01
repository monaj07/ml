"""
Utility codes for splitting the images into Train and Test partitions:
"""
import os
import numpy as np
import random
import shutil

# base_path = os.getcwd()
# data_path = os.path.join(base_path, "data/CaltechFaces/NoFace")
# categories = os.listdir(data_path)
# test_path = os.path.join(base_path, "data/CaltechFaces/Test/NoFace")
# for cat in categories:
#     if os.path.isdir(os.path.join(data_path, cat)):
#         image_files = os.listdir(os.path.join(data_path, cat))
#         choices = np.random.choice([0, 1], size=(len(image_files),), p=[.7, .3])
#
#         for idx, file in enumerate(image_files):
#             if choices[idx]:
#                 origin_path = os.path.join(data_path, cat, file)
#                 dest_dir = os.path.join(test_path, cat)
#                 dest_path = os.path.join(test_path, cat, file)
#                 if not os.path.isdir(dest_dir):
#                     os.mkdir(dest_dir)
#                 shutil.move(origin_path, dest_path)
# ######################################################################

# base_path = os.getcwd()
# data_path = os.path.join(base_path, 'data', 'lfw', 'data')
# persons = os.listdir(data_path)
# people_dict = {}
# pos_pairs = {}
# for person in persons:
#     if os.path.isdir(os.path.join(data_path, person)):
#         samples = os.listdir(os.path.join(data_path, person))
#         if len(samples) > 1:
#             people_dict[person] = samples
#             pos_pairs[person] = len(samples) * (len(samples) - 1) // 2
#
# for k, v in people_dict.items():
#     rnd = random.choice([1, 0])
#     if rnd:
#         shutil.copytree(os.path.join(data_path, k), os.path.join(data_path, 'train', k))
#     else:
#         shutil.copytree(os.path.join(data_path, k), os.path.join(data_path, 'test', k))
# print(people_dict)


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.features)

    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.project.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
