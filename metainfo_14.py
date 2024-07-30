import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.utils.data as data

import os
import re
import csv
import json
import torch
import tarfile
import pickle
import numpy as np
import pandas as pd
import random
random.seed(2021)
from PIL import Image
from scipy import io as scio
from math import radians, cos, sin, asin, sqrt, pi

import pickle

## PRAM
# revised_txt = "_plantae"
revised_txt = "_Mammalia"
save_pickle_path = '/GPFS/public/iNaturalist/processed/'
root = '/GPFS/public/iNaturalist'
dataset = 'inaturelist2018'
istrain = True


def self_defined_load_file(root, dataset, istrain=True, meta_info=True):
    """ This module is made for extracting information with meta_info, and probably can be embedded in other modules"""

    if dataset == 'inaturelist2018':
        year_flag = 8

    """
    Each dictinary following includes:
    1.loc_uncert: an int number
    2.data: a string in the form of yyyy-mm-dd
    3.valid:True/False
    4.user_id: numbers
    5.lat: float
    6.date_c:float
    7.lon: float
    8.id: int
    """

    """ Map_2018 construction: a dict "id":"name" """
    with open(os.path.join(root, f'categories{revised_txt}.json'), 'r') as f:
        map_label = json.load(f)
    map_2018 = dict()
    for _map in map_label:
        map_2018[int(_map['id'])] = _map['name'].strip().lower()
    
    """ Validation set meta info as a dictionary """
    with open(os.path.join(root, f'val201{year_flag}_locations{revised_txt}.json'), 'r') as f:
        val_location = json.load(f)
    val_id2meta = dict()
    for meta_info in val_location:
        val_id2meta[meta_info['id']] = meta_info

    """ Training set meta info as a dictionary """
    with open(os.path.join(root, f'train201{year_flag}_locations{revised_txt}.json'), 'r') as f:
        train_location = json.load(f)
    train_id2meta = dict()
    for meta_info in train_location:
        train_id2meta[meta_info['id']] = meta_info

    """ Validation set as a dictionary """
    with open(os.path.join(root, f'val201{year_flag}{revised_txt}.json'), 'r') as f:
        val_class_info = json.load(f)

    """ Training set as a dictionary """
    with open(os.path.join(root, f'train201{year_flag}{revised_txt}.json'), 'r') as f:
        train_class_info = json.load(f)

    """ A list of all names"""
    categories_2018 = [x['name'].strip().lower() for x in map_label]

    """ dict() between class and idx """
    """ Specify the index of each class"""
    class_to_idx = {c: idx for idx, c in enumerate(categories_2018)}

    id2label = dict()
    for categorie in val_class_info['categories']:
        name = map_2018[int(categorie['name'])]
        id2label[int(categorie['id'])] = name.strip().lower()

    class_info = train_class_info if istrain else val_class_info
    id2meta = train_id2meta if istrain else val_id2meta
    images_and_targets = []
    if meta_info:
        temporal_info = []
        spatial_info = []
    for image, annotation in zip(class_info['images'], class_info['annotations']):
        file_path = os.path.join(root, image['file_name'])
        id_name = id2label[int(annotation['category_id'])]
        target = class_to_idx[id_name]
        image_id = image['id']
        date = id2meta[image_id]['date']
        latitude = id2meta[image_id]['lat']
        longitude = id2meta[image_id]['lon']
        location_uncertainty = id2meta[image_id]['loc_uncert']
        if meta_info:
            temporal_info = get_temporal_info(date, miss_hour=True)
            spatial_info = get_spatial_info(latitude, longitude)
            images_and_targets.append({"path":file_path, "label":target, "metainfo": temporal_info+spatial_info})
        else:
            images_and_targets.append({"path":file_path, "label":target})
    # print(images_and_targets[0])
    # print(np.array(images_and_targets[-1]['metainfo']).shape)
    # print(type(images_and_targets[-1]['metainfo']))
    return images_and_targets


def get_spatial_info(latitude, longitude):
    """ This module helps to extract the exact spatial info"""
    """ Input: floats/ Outputs: A list [x = cos(latitude)*cos(longitude),y = cos(latitude)*sin(longitude),z = sin(latitude)]"""
    if latitude and longitude:
        '''
        latitude = radians(latitude)
        longitude = radians(longitude)
        x = cos(latitude)*cos(longitude)
        y = cos(latitude)*sin(longitude)
        z = sin(latitude)
        '''
        # result to 0-1
        x = (latitude + 90) / 180
        y = (longitude + 360) / 360
        return [x, y]
        
        # return [x, y, z]
    else:
        return [0, 0]
        # return [0, 0, 0]


def get_temporal_info(date, miss_hour=False):
    """ Note that since using 2018 version, miss_hour is always false"""
    """ This module helps to extract the exact temporal info"""
    """ Input: a string/ Output: A list [x_month,y_month,x_hour,y_hour]"""
    try:
        if date:
            if miss_hour:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*)', re.I)
            else:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*) (\d*):(\d*):(\d*)', re.I)
            m = pattern.match(date.strip())

            if m:
                year = int(m.group(1))
                month = int(m.group(2))
                day = int(m.group(3))
                x_month = sin(2*pi*month/12)
                y_month = cos(2*pi*month/12)
                '''
                if month == 1: 
                    rst = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                if month == 2: 
                    rst = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                if month == 3: 
                    rst = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                if month == 4: 
                    rst = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                if month == 5: 
                    rst = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                if month == 6: 
                    rst = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                if month == 7: 
                    rst = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                if month == 8: 
                    rst = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                if month == 9: 
                    rst = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                if month == 10: 
                    rst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                if month == 11: 
                    rst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                if month == 12: 
                    rst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                '''

                if miss_hour:
                    x_hour = 0
                    y_hour = 0
                else:
                    hour = int(m.group(4))
                    x_hour = sin(2*pi*hour/24)
                    y_hour = cos(2*pi*hour/24)
                # return rst
                return [x_month, y_month]
            else:
                return[0,0]
                # return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            return[0,0]
            # return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    except:
        return[0,0]
        # return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def load_file(root, dataset):
    if dataset == 'inaturelist2017':
        year_flag = 7
    elif dataset == 'inaturelist2018':
        year_flag = 8

    if dataset == 'inaturelist2018':
        with open(os.path.join(root, f'categories{revised_txt}.json'),'r') as f:
            map_label = json.load(f)
        map_2018 = dict()
        for _map in map_label:
            map_2018[int(_map['id'])] = _map['name'].strip().lower()

    with open(os.path.join(root, f'val201{year_flag}_locatioins{revised_txt}.json'), 'r') as f:
        val_location = json.load(f)
    val_id2meta = dict()
    for meta_info in val_location:
        val_id2meta[meta_info['id']] = meta_info

    with open(os.path.join(root, f'train201{year_flag}_locations{revised_txt}.json'), 'r') as f:
        train_location = json.load(f)
    train_id2meta = dict()
    for meta_info in train_location:
        train_id2meta[meta_info['id']] = meta_info

    with open(os.path.join(root, f'val201{year_flag}{revised_txt}.json'), 'r') as f:
        val_class_info = json.load(f)
    with open(os.path.join(root, f'train201{year_flag}{revised_txt}.json'), 'r') as f:
        train_class_info = json.load(f)

    if dataset == 'inaturelist2017':
        categories_2017 = [x['name'].strip().lower() for x in val_class_info['categories']]
        class_to_idx = {c: idx for idx, c in enumerate(categories_2017)}
        id2label = dict()
        for categorie in val_class_info['categories']:
            id2label[int(categorie['id'])] = categorie['name'].strip().lower()
    elif dataset == 'inaturelist2018':
        categories_2018 = [x['name'].strip().lower() for x in map_label]
        class_to_idx = {c: idx for idx, c in enumerate(categories_2018)}
        id2label = dict()
        for categorie in val_class_info['categories']:
            name = map_2018[int(categorie['name'])]
            id2label[int(categorie['id'])] = name.strip().lower()
    print(train_class_info)
    return train_class_info, train_id2meta, val_class_info, val_id2meta, class_to_idx, id2label

if __name__ == '__main__':
    preprocessed_metainfo = self_defined_load_file(root, dataset, istrain=istrain, meta_info=True)
    # print(preprocessed_metainfo)
    
    if istrain:
        f = open(save_pickle_path + 'train_' + 'metainfo' + revised_txt + '4.pkl', 'wb')       
    else:
        f = open(save_pickle_path + 'val_' + 'metainfo' + revised_txt + '4.pkl', 'wb')
    pickle.dump(preprocessed_metainfo, f)
    f.close()
    

