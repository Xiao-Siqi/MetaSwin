import os
import numpy as np
import cv2
import torch

import torchvision.transforms as transforms
from PIL import Image
import copy
import torch

import pickle
import pandas

from .randaug import RandAugment


def build_loader(args):
    train_set, train_loader = None, None
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None
    if args.val_root is not None:
        val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        print(root)
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root):
        data_infos = []
        # folders = os.listdir(root)
        # availf=[]
        # for f in folders:
        #     if (os.path.isdir(os.path.join(root,f))):
        #         availf.append(f)
        # #print(availf)
        # availf.sort() # sort by alphabet
        # print("[dataset] class number:", len(availf))
        # for class_id, folder in enumerate(availf):
        #     #print(root+folder)
        #     files = os.listdir(root+folder)
        #     #print(class_id)
        #     for file in files:
        #         if (file[0:2]!="._"):
        #             data_path = root+folder+"/"+file
        #             #print(file[0:2])
        #             data_infos.append({"path":data_path, "label":class_id})
        #         #print(data_infos)
        
        f = open(root, 'rb')
        data_infos = pickle.load(f)

        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        meta_info = self.data_infos[index]["metainfo"]
        meta_info = torch.Tensor(meta_info)
        
        
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.

        
        
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)

        # C H W 
        c,h,w = img.shape

        # process the metainfo
        meta_info = meta_info.reshape(1,-1)
        # F.pad(original_values, pad=(1,0,0,0,0,0), mode="constant",value=0)  
        m = torch.nn.ZeroPad2d((0,w-meta_info.shape[1],0,0))
        meta_info = m(meta_info).reshape(1,-1, 1)
        meta_info = meta_info.repeat(3,1,1)
        img = torch.cat([img,meta_info], axis=2)
        # print(img.shape)
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label
