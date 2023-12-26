import os
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
import torch.utils.data as data
import torchvision.transforms as transforms


class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def folder_content_getter(folder_path,im_curve):

    cate_names = list(np.sort(os.listdir(folder_path)))

    if 'domainnet' in folder_path.lower():
        cate_names = ['airplane','ambulance','apple','backpack','banana','bathtub','bear','bed','bee','bicycle','bird','book','bridge','bus',
        'butterfly','cake','calculator','camera','car','cat','chair','clock','cow','dog','dolphin','donut','drums','duck','elephant',
        'fence','fork','horse','house','rabbit','scissors','sheep','strawberry','table','telephone','truck']

        image_path_list = []
        image_cate_list = []

        for cate_name in cate_names:
            sub_folder_path = os.path.join(folder_path, cate_name)
            if os.path.isdir(sub_folder_path):
                image_names = list(np.sort(os.listdir(sub_folder_path)))
                for image_name in image_names:
                    image_path = os.path.join(sub_folder_path, image_name)
                    image_path_list.append(image_path)
                    image_cate_list.append(cate_names.index(cate_name))

    
    else:    
        image_path_list = []
        image_cate_list = []

        temp = 0
        cls_dict =dict()
        cate_dict = dict()

        for cate_name in cate_names:
            sub_folder_path = os.path.join(folder_path, cate_name)
            if os.path.isdir(sub_folder_path):
                image_names = list(np.sort(os.listdir(sub_folder_path)))
                image_path_list = []
                image_cate_list = []
                for image_name in image_names:
                    image_path = os.path.join(sub_folder_path, image_name)
                    image_path_list.append(image_path)
                    image_cate_list.append(cate_names.index(cate_name))
                
                cls_dict[temp] = image_path_list
                cate_dict[temp] = image_cate_list
                temp += 1

        imb_num_list = get_img_num_per_cls(cls_dict, temp, 'exp', 0.1, im_curve=im_curve)
        image_path_list, image_cate_list = gen_imbalanced_data(imb_num_list, cls_dict,cate_dict, temp)
        # img_num_list = get_img_num_per_cls_beta(cls_dict, temp, alpha=alpha, beta=beta)
        # image_path_list, image_cate_list = gen_imbalanced_data(img_num_list, cls_dict,cate_dict, temp)
    

    return image_path_list, image_cate_list

def get_img_num_per_cls(cls_dict, cls_num, imb_type, imb_factor, im_curve='RS'):
    total_img = 0
    for k in cls_dict.keys():
        total_img += len(cls_dict[k])
    img_max = total_img / cls_num
    img_num_per_cls = []

    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)

    if im_curve == 'UT':
        img_num_per_cls.reverse()

    return img_num_per_cls

# def get_img_num_per_cls_beta(cls_dict, cls_num, alpha=2.0, beta=2.0):
#     total_img = 0
#     for k in cls_dict.keys():
#         total_img += len(cls_dict[k])
#     img_max = total_img / cls_num
#     img_num_per_cls = []

#     b = beta
#     a = alpha

#     x = np.arange(0.0, 1.0, 1.0/cls_num) + 1.0 / cls_num * 0.5
#     y = scipy.stats.beta.pdf(x, a, b)
#     p = y / y.sum()

#     for cls_idx in range(cls_num):
#         num = max(img_max * p[cls_idx], 1)
#         img_num_per_cls.append(int(num))

#     return img_num_per_cls


def gen_imbalanced_data(img_num_per_cls, cls_dict,cate_dict, cls_num):
    new_lines = []
    new_cates = []

    classes = range(cls_num)

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        if the_img_num <= len(cls_dict[the_class]):
            pick_idx = random.sample(list(range(len(cls_dict[the_class]))), the_img_num)
        else:
            pick_idx = random.choices(list(range(len(cls_dict[the_class]))), k=the_img_num)

        for i in pick_idx:
            new_lines.append(cls_dict[the_class][i])
            new_cates.append(cate_dict[the_class][i])

    return new_lines,new_cates



class EvalDataset(data.Dataset):
    def __init__(self,
                 datasetA_dir,im_curve):

        self.datasetA_dir = datasetA_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalize,
                     ])

        self.image_paths_A, self.image_cates_A = folder_content_getter(datasetA_dir, im_curve)
        
        self.domainA_size = len(self.image_paths_A)
        
    def __getitem__(self, index):

        index_A = np.mod(index, self.domainA_size)
        
        image_path_A = self.image_paths_A[index_A]
        
        image_A = self.transform(Image.open(image_path_A).convert('RGB'))
        
        target_A = self.image_cates_A[index_A]
        
        return image_A, target_A, index_A

    def __len__(self):

        return self.domainA_size

class TrainDataset(data.Dataset):
    def __init__(self,
                 datasetA_dir, im_curve):

        self.datasetA_dir = datasetA_dir
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        
        self.transform =  transforms.Compose([transforms.Resize(256),
                  transforms.RandomResizedCrop(224),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize])
       
    

        self.image_paths_A, self.image_cates_A = folder_content_getter(datasetA_dir, im_curve)
        
        self.domainA_size = len(self.image_paths_A)
        
    def __getitem__(self, index):

        if index >= self.domainA_size:
            index_A = random.randint(0, self.domainA_size - 1)
        else:
            index_A = index

       

        image_path_A = self.image_paths_A[index_A]
        
        x_A = Image.open(image_path_A).convert('RGB')
        q_A = self.transform(x_A)
        

        target_A = self.image_cates_A[index_A]
        
        return q_A, target_A, index_A

    def __len__(self):

        return self.domainA_size


def set_model_self(source_loader, target_loader, target_loader_val, model_self, target_loader_test=None, source_loader_test=None):
    source_loader.mode_self = model_self
    target_loader.mode_self = model_self
    target_loader_val.mode_self = model_self

    if target_loader_test:
        target_loader_test.mode_self = model_self
        source_loader_test.mode_self = model_self

