import torch.utils.data as data

from PIL import Image

import xml.etree.ElementTree as ET
import numpy as np
import os
import os.path
import sys


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)







class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, dataset_list, loader, extensions, transform=None, target_transform=None):
        self.num_classes = 20
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

       #classes, class_to_idx = self._find_classes(root)
        samples = self.make_dataset(root, dataset_list, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

        self.dataset_list = dataset_list
        
    def make_dataset(self, dir, dataset_list, extensions):
        images = []
        cls = []
        dir = os.path.expanduser(dir)
        
        anno_path = dir.replace('JPEGImages','Annotations')
        with open(dataset_list, 'r') as f:
            li = f.readlines()
            for l in li :
                name = l.replace('\n','')
                images.append(os.path.join(dataset_list, name + '.jpg'))
                cls.append(self.get_cls(anno_path, name))
        return (images, cls)



    def get_cls(self, anno_path, index):
        """
        Load image labels.
        """
        filename = os.path.join(anno_path, index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
#        if not self.config['use_diff']:
#            # Exclude the samples labeled as difficult
#            non_diff_objs = [
#                obj for obj in objs if int(obj.find('difficult').text) == 0]
#            # if len(non_diff_objs) != len(objs):
#            #     print 'Removed {} difficult objects'.format(
#            #         len(objs) - len(non_diff_objs))
#            objs = non_diff_objs
        num_objs = len(objs)

        gt_classes = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            gt_classes[ix] = cls

        real_label = np.zeros(self.num_classes).astype(np.float32)
        # TODO Fill this part
        for label in gt_classes:
            real_label[label] = 1
        return real_label

#    def _find_classes(self, dir):
#        """
#        Finds the class folders in a dataset.
#        Args:
#            dir (string): Root directory path.
#        Returns:
#            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#        Ensures:
#            No class is a subdirectory of another.
#        """
#        if sys.version_info >= (3, 5):
#            # Faster and available in Python 3.5 and above
#            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#        else:
#            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#        classes.sort()
#        class_to_idx = {classes[i]: i for i in range(len(classes))}
#        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        pdb.set_trace()
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, dataset_list=None, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, dataset_list, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
