import sys, os
import yaml
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    img_list = [f for f in os.listdir("./dataset/chroma/test") if not f.startswith('.')]

    outputs_dir = os.path.join(os.path.dirname(args.model_path), "test")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
        print("Directory " , outputs_dir ,  " Created ")
    else:
        print("Directory " , outputs_dir ,  " already exists")


    for img_name in img_list:
        img_path = "./dataset/chroma/test/" + img_name

        img = np.loadtxt(img_path, dtype=np.float16)
        #img = np.reshape(img, (3, 360, 480))
        img = np.reshape(img, (172800, 3))
        Cb = img[:, 1]
        Cr = img[:, 2]
        img1 = np.reshape(img[:,0], (360, 480))
        img2 = np.array([img1, img1, img1])
        #img[1, :, :] = img[0, :, :]
        #img[2, :, :] = img[0, :, :]

        max_val = np.array([512.0])
        img2 -=  max_val

        # normalize the data
        img2 /= (2*max_val)

        img2 = torch.from_numpy(img2).float()
        img2 = img2.unsqueeze(0)
        #print(model_name)
        # Setup Model
        model_dict = { "arch" : model_name }
        model = get_model(model_dict, n_classes = 3, version=args.dataset)
        state = convert_state_dict(torch.load(args.model_path)["model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(device)

        images = img2.to(device)
        #outputs = img2.to(device)
        outputs = model(images)

        if args.dcrf:
            unary = outputs.data.cpu().numpy()
            unary = np.squeeze(unary, 0)
            unary = -np.log(unary)
            unary = unary.transpose(2, 1, 0)
            w, h, c = unary.shape
            unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
            unary = np.ascontiguousarray(unary)

            resized_img = np.ascontiguousarray(resized_img)

            d = dcrf.DenseCRF2D(w, h, loader.n_classes)
            d.setUnaryEnergy(unary)
            d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

            q = d.inference(50)
            mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
            decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
            dcrf_path = args.out_path[:-4] + "_drf.png"
            misc.imsave(dcrf_path, decoded_crf)
            print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        if model_name in ["pspnet", "icnet", "icnetBN"]:
            pred = pred.astype(np.float32)
            # float32 with F mode, resize back to orig_size
            pred = misc.imresize(pred, orig_size, "nearest", mode="F")

        # save outputs
        outputs = torch.squeeze(outputs, 0)
        outputs = outputs.detach().numpy()
        outputs1 = np.array([outputs[0].flatten(), outputs[1].flatten(), outputs[2].flatten()]).transpose()
        outputs1 *= (2*max_val)
        outputs1 += max_val
        outputs2 = np.array([outputs1[:,0].transpose(), Cb.transpose(), Cr.transpose()]).transpose()
        np.savetxt(outputs_dir + "/" + img_name, outputs2, fmt='%s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="chroma",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default=None,
        help="Path of the output segmap",
    )
    args = parser.parse_args()
    test(args)
