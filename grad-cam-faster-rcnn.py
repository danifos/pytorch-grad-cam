import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
sys.path.append('/home/user/deep-learning/my-faster-rcnn')
from lib.faster_r_cnn import FasterRCNN
from lib.consts import voc_names
import numpy as np
import argparse

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.CNN, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        RPN_cls, RPN_reg = self.model.RPN(output)
        roi_scores, roi_coords, proposals = self.model.test_RCNN(x, {'scale': (1,)}, output,
                                                                 RPN_cls, RPN_reg, fetch_tensors=True)

        return target_activations, (RPN_cls, RPN_reg, roi_scores, roi_coords, proposals)

def preprocess_image(img):
    img = np.float32(img)
    means=[122.7717, 115.9465, 102.9801]

    preprocessed_img = img.copy()
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask, proposal, shape):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    # cam = cv2.resize(img, shape)
    cv2.imwrite("cam.jpg", cam)
    import matplotlib.pyplot as plt
    plt.imshow(cam[:,:,::-1])
    bbox = proposal
    plt.gca().add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2], bbox[3], fill=False)
    )
    plt.show()

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        RPN_cls, RPN_reg, roi_scores, roi_coords, proposals = output
        output = roi_scores
        output_np = output.cpu().data.numpy()
        output_np[:, 0] = 0

        if index == None:
            index = np.unravel_index(np.argmax(output_np), output_np.shape)
            proposal = proposals[:, index[0]]
        else:
            indices = np.unravel_index(np.argsort(output_np.reshape(-1)), output_np.shape)
            index = tuple(i[index] for i in indices)
        print("Class:", voc_names[index[1]])
        proposal = proposals[:, index[0]]

        one_hot = np.zeros_like(output_np, dtype=np.float32)
        one_hot[index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.CNN.zero_grad()
        self.model.RPN.zero_grad()
        self.model.RCNN.zero_grad()
        one_hot.backward()

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (600, 600))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, proposal

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input),
                                                 grad_output, positive_mask_1),
                                   positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # TODO: Start from here

        # replace ReLU with GuidedBackpropReLU
        for child in self.model.children():
            for idx, module in child._modules.items():
                if module.__class__.__name__ == 'ReLU':
                    child._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward()

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_false', default=True,
                        help='Do not use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--index', type=str, default='None')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(
        model=FasterRCNN('/home/user/deep-learning/my-faster-rcnn/results/result-31-n/param-16-80176.pth'),
        target_layer_names = ["29"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    shape = img.shape[:2]
    img = cv2.resize(img, (600, 600))
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = eval(args.index)

    mask, proposal = grad_cam(input, target_index)

    show_cam_on_image(img, mask, proposal, shape)

    # gb_model = GuidedBackpropReLUModel(
    #     model=FasterRCNN('/home/user/deep-learning/my-faster-rcnn/results/result-31-n/param-16-80176.pth'),
    #     use_cuda=args.use_cuda)
    # gb = gb_model(input, index=target_index)
    # utils.save_image(torch.from_numpy(gb), 'gb.jpg')
    
    # cam_mask = np.zeros(gb.shape)
    # for i in range(0, gb.shape[0]):
    #     cam_mask[i, :, :] = mask
    
    # cam_gb = np.multiply(cam_mask, gb)
    # utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
