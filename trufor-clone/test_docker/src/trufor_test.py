# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""

import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
from torch.nn import functional as F

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from config import update_config
from config import _C as config
from data_core import myDataset

parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default='../output', help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

input = args.input
output = args.output
gpu = args.gpu
save_np = args.save_np

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

# Check if the input is a wildcard pattern, a file, or a directory
if '*' in input:
    # If input is a wildcard pattern, use glob to find all matching files
    list_img = glob(input, recursive=True)
    # Filter out directories from the list
    list_img = [img for img in list_img if not os.path.isdir(img)]
elif os.path.isfile(input):
    # If input is a file, create a list with just that file
    list_img = [input]
elif os.path.isdir(input):
    # If input is a directory, use glob to find all files in the directory and its subdirectories
    list_img = glob(os.path.join(input, '**/*'), recursive=True)
    # Filter out directories from the list
    list_img = [img for img in list_img if not os.path.isdir(img)]
else:
    # If input is neither a file nor a directory, raise an error
    raise ValueError("input is neither a file or a folder")

# Create a dataset from the list of images
test_dataset = myDataset(list_img=list_img)

# Create a DataLoader for the dataset
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1)  # 1 to allow arbitrary input sizes

# Check if a model file is specified in the configuration
if config.TEST.MODEL_FILE:
    model_state_file = config.TEST.MODEL_FILE
else:
    # If no model file is specified, raise an error
    raise ValueError("Model file is not specified.")

# Load the model from the specified file
print('=> loading model from {}'.format(model_state_file))
checkpoint = torch.load(model_state_file, map_location=torch.device(device))

# Check the model name in the configuration and create the corresponding model
if config.MODEL.NAME == 'detconfcmx':
    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    model = confcmx(cfg=config)
else:
    # If the model name is not recognized, raise an error
    raise NotImplementedError('Model not implemented')

# Load the model state from the checkpoint
model.load_state_dict(checkpoint['state_dict'])
# Move the model to the specified device
model = model.to(device)

# # print(model)
# # write model into a text file
# with open('model.txt', 'w') as f:
#     print(model, file=f)


# Loop over the data in the DataLoader
with torch.no_grad():
    for index, (rgb, path) in enumerate(tqdm(testloader)):
        print(input)
        print(path)
        # Determine the output filename based on the input and output parameters
        if os.path.splitext(os.path.basename(output))[1] == '':  # output is a directory
            path = path[0]
            root = input.split('*')[0]

            if os.path.isfile(input):
                sub_path = path.replace(os.path.dirname(root), '').strip()
            else:
                sub_path = path.replace(root, '').strip()

            if sub_path.startswith('/'):
                sub_path = sub_path[1:]

            filename_out = os.path.join(output, sub_path) + '.npz'
        else:  # output is a filename
            filename_out = output

        if not filename_out.endswith('.npz'):
            filename_out = filename_out + '.npz'

        # If the output file does not exist, process the image and save the results
        if not (os.path.isfile(filename_out)):
            try:
                # Move the image to the specified device
                rgb = rgb.to(device)
                # Set the model to evaluation mode
                model.eval()

                # Initialize the variables for the detection, confidence, and NPP results
                det = None
                conf = None

                # Run the model on the image
                pred, conf, det, npp = model(rgb)

                # Process the results and convert them to numpy arrays
                # Check if the confidence tensor is not None
                if conf is not None:
                    # Remove dimensions of size 1 from the confidence tensor
                    conf = torch.squeeze(conf, 0)
                    # Apply the sigmoid function to the confidence tensor to normalize its values between 0 and 1
                    conf = torch.sigmoid(conf)[0]
                    # Move the confidence tensor from the GPU to the CPU and convert it to a numpy array
                    conf = conf.cpu().numpy()

                # Check if the NPP tensor is not None
                if npp is not None:
                    # Remove dimensions of size 1 from the NPP tensor
                    npp = torch.squeeze(npp, 0)[0]
                    # Move the NPP tensor from the GPU to the CPU and convert it to a numpy array
                    npp = npp.cpu().numpy()

                # Check if the detection tensor is not None
                if det is not None:
                    # Apply the sigmoid function to the detection tensor to normalize its value between 0 and 1
                    # and convert it to a Python scalar
                    det_sig = torch.sigmoid(det).item()

                # Remove dimensions of size 1 from the prediction tensor
                pred = torch.squeeze(pred, 0)
                # Apply the softmax function to the prediction tensor to normalize its values between 0 and 1
                # and select the second channel (index 1)
                pred = F.softmax(pred, dim=0)[1]
                # Move the prediction tensor from the GPU to the CPU and convert it to a numpy array
                pred = pred.cpu().numpy()

                # Create a dictionary to store the results
                out_dict = dict()
                # Store the prediction map in the dictionary
                out_dict['map'] = pred
                # Store the size of the input image in the dictionary
                out_dict['imgsize'] = tuple(rgb.shape[2:])
                # If the detection tensor is not None, store the detection score in the dictionary
                if det is not None:
                    out_dict['score'] = det_sig
                # If the confidence tensor is not None, store the confidence map in the dictionary
                if conf is not None:
                    out_dict['conf'] = conf
                # If the save_np flag is True, store the NPP map in the dictionary
                if save_np:
                    out_dict['np++'] = npp

                # Create the output directory if it does not exist
                from os import makedirs
                makedirs(os.path.dirname(filename_out), exist_ok=True)

                # Save the results to the output file
                np.savez(filename_out, **out_dict)
            except:
                # If an error occurs, print the traceback and continue with the next image
                import traceback
                traceback.print_exc()
                pass

