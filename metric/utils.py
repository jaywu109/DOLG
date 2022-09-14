import os
import torch
from PIL import Image

IMG_FORMATS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.jfif', '.webp', '.JPEG'}


def find_image(root):
    for folder, _, files in os.walk(root, followlinks=True):
        for f in files:
            _, suffix = os.path.splitext(f)
            if suffix in IMG_FORMATS:
                yield os.path.basename(folder), os.path.join(folder, f)
            else:
                print('Cannot process {}'.format(f))


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def freeze_weight(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_weight(module):
    for param in module.parameters():
        param.requires_grad = True


class RGB2BGR(object):
    """
    Converts a PIL image from RGB to BGR
    """

    def __init__(self):
        pass

    def __call__(self, img):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
