#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/Final Project Lab/Image Classification - Predicting Function 
#                                                                             
# PROGRAMMER: Toyin O.
# DATE CREATED: 07/17/2018                                  
# 
# PURPOSE: Train FeedForward Network to classify flowers from  102 categories

##

# Import all the necessary packages and modules 
import argparse
import numpy as np
import torch
import json
from PIL import Image
from torch import nn
import torch.nn.functional as F


# Main program function defined here
def main():
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
    ###Extract all supplied CL arguments and create variables
    chk_path = in_arg.chk_path
    imgpath = in_arg.imgpath
    topk = in_arg.topk
    cat_file = in_arg.cat_file
    gpu = in_arg.gpu
    
    ##Set the training device to Cuda if available else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu ==0 else "cpu")
    #Load Checkpoint Dictionary
    checkpnt = load_checkpoint(chk_path)
    arch = checkpnt['arch'] 
    ## Create Model from checkpoint 
    model, input_units = get_models(arch)
    
    ## Extract Other parameters from checkpoint 
    hidden_units = checkpnt['hidden_units']
    output_units = checkpnt['output_units']
    epochs = checkpnt['epochs']
    
    #Define Classifier
    classifier = net_classifier(arch, hidden_units, output_units, device,model, input_units)   
    # Freeze model parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
     
    ###Load model parameters
    if 'resnet' in arch:
        model.fc = classifier
    else:
        model.classifier = classifier
        
    model.load_state_dict(checkpnt['state_dict'])
    model.class_to_idx = checkpnt['class_to_idx']
    
    ###Predict Top Probabilities/Categories for Image file
    probs, classes = predict(imgpath, model, gpu, topk)
    print(probs)
    print(classes)

    mappings = category_to_name( probs, classes, cat_file)
    print("\nThe top {} predictions for image {} are : ".format(topk,imgpath))
    #for i, (a, b) in enumerate(zip(alist, blist)):
    for i, (cat, prob, class_id) in enumerate(zip(mappings, probs, classes), 1):
        print("{}.  {} : in Class: {} and with a probability of {:.2f}%\n".format(i, cat.title(),class_id, prob*100))
    
def load_checkpoint(filepath):
    """
    Retrieves and loads saved checkpoint file.   ArgumentParser object.
    Parameters:
     Filepath - location of saved checkpoint file
    Returns:
     checkpoint - dictionary containing saved network parameters
    """
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    return checkpoint
    

def process_image(image_loc):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        Parameters:
          image_loc - location of image file to be classified
        Returns:
          final_image - (Numpy) array representation of image file
    '''
    #Sets target image resolution
    target = 224
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = Image.open(image_loc)
    # Resize to 256 
    width, height = image.size   
    ratio = int(width)/int(height)
    
    if width <= height:
        image = image.resize((256, int(256/ratio)))
    else:
        image = image.resize((int(256/ratio), 256))
    # Crop
    width, height = image.size   # Get dimensions
    left = (width - target)/2
    top = (height - target)/2
    right = (width + target)/2
    bottom = (height + target)/2
    image = image.crop((left, top, right, bottom))
    
    #Normlize
    np_image = np.array(image)
    image_norm = (np_image/255 - mean) / std
    
    #Transpose for Pytorch
    final_image = image_norm.transpose((2, 0, 1))
    
    return final_image

def predict(image_path, model, gpu, topk):
    ''' Returns the top 'k' predictions/classifications of the input image
        Parameters:
          image_path - location of image file to be classified
          model - CNN used to classify the image
          device - processor type - GPU or CPU 
          topk - returns topk probabilities/classifications of input image
        Returns:
          probs - Top 'k' numeric probabilities of image
          top_label - Top 'k' category IDs
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu ==0 else "cpu")
    # Import Image as Numpy Array
    img = process_image(image_path)
    #Convert from Numpy to Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    #Add a unit dimension to the Tensor
    image = image_tensor.unsqueeze(0)
    #Move tensors/model to CPU
    model, image = model.to(device), image.to(device)
    ##Turn on Eval
    model.eval()
    model.requires_grad = False
    # Probs
    prob = torch.exp(model.forward(image))
    top_probs, top_labels = prob.topk(topk)
    
    probs, labels = top_probs.data.cpu().numpy()[0], top_labels.data.cpu().numpy()[0] 
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    top_label = [idx_to_class[labels[i]] for i in range(labels.size)]
    
    return probs, top_label
    
# Creates function to receive user input from argparse function
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser(description='Train the Network using Transfer Learning to Classify Flowers')
    
    # Creates 5 command line arguments 
    ## args.chk_path full path of checkpoint file
    ## args.imgpath for the path of a single image to be classified
    ## args.topk for the top 'k' predictions/classifications for the image passed
    ## args.gpu settings allows the user to choose whether to train with GPU (setting = 0) or CPU (setting < 0)
    ## args.cat_file - specificies location of json file containing ID to category mapping
    
    parser.add_argument('--chk_path', '-p', type=str, default='data/checkpoint.pth', required=True,
                        help='Full path to directory containing saved checkpoint file ')
    parser.add_argument('--imgpath','-f',type=str, default='../aipnd-project/flowers/test/30/image_03482.jpg', help='full file path to single image being                               tested')
    parser.add_argument('--topk', '-topk', type=int, default=1,help='Number of likely classes to return')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU, 0 for GPU)')
    parser.add_argument('--cat_file', '-c', type=str, default= 'cat_to_name.json',
                        help='specifies file path of JSON file containing category names')
                        
    # returns parsed argument collection
    return parser.parse_args()


def net_classifier(arch, hidden_units, output_units, device, model, input_units):
    """
    A new feedforward network is defined for use as a classifier using the features as input...
    
    Parameters:
    arch - Pretrained CNN model as input - e,g VGG13
    hidden_units - No of hidden units
    output_units - number of output units
    dropout - dropout ratio 
    gpu - GPU State 
    device - Device State - GPU or CPU 
    Returns:
    model.classifier, model - Returns the model and model.classifier
    
    """
    #turn dropoout off for inference
    dropout = 1.0
    #model,input_units = get_models(arch)
    model.to(device)
    print("Loading Model Using {}".format(device))
    
    #Define Classifier
    class Network(nn.Module):
        def __init__(self, input_size,output_size, hidden_layers, dropout):
            super().__init__()
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            self.output = nn.Linear(hidden_layers[-1], output_size)
            self.dropout = nn.Dropout(dropout)
        ## Forward function
        def forward(self, x):
            for each in self.hidden_layers:
                x = F.relu(each(x))
                #x = self.dropout(x)
            x = self.output(x)
            return F.log_softmax(x, dim=1)
    
    ##Instatiate the classifier function
    classifier = Network(input_units, output_units, hidden_units, dropout)
    
    return classifier 
        
def get_models(arch):
    """
    Receives the pretrained CNN model from torchvision.models as input and freezes the parameters...
    
    Parameters:
     arch - Receives the pretrained CNN model as input - e,g VGG13. Function processes valid VGG and Densenet models only(string)
    Returns:
     model, input_units - Returns the model of CNN and the number of input units expected by that model
    """
    
    from torchvision import datasets, transforms, models
    if 'vgg' in arch:
        valid = [11,13,16,19]
        arch_units =  ''.join(i for i in arch if i.isdigit())
        if int(arch_units) in valid: 
            model = "models"+"."+arch+"(pretrained=True)"
            model = eval(model)
            input_units = model.classifier[0].in_features
        else:
            print('Unrecognized Model!!! Please pass a valid Model')
    elif 'densenet' in arch:
        valid = [121,169,201,161]
        arch_units =  ''.join(i for i in arch if i.isdigit())
        if int(arch_units) in valid: 
            model = "models"+"."+arch+"(pretrained=True)"
            model = eval(model)
            input_units = model.classifier.in_features
        else:
            print('Unrecognized Model!!! Please pass a valid Model')    
    elif 'resnet' in arch:
        valid = [18,34,50,101,152]
        arch_units =  ''.join(i for i in arch if i.isdigit())
        if int(arch_units) in valid: 
            model = "models"+"."+arch+"(pretrained=True)"
            model = eval(model)
            input_units = model.fc.in_features
        else:
            print('Unrecognized Model!!! Please pass a valid Model')    
    else:
        print('Unexpected network architecture', model)
    
    return model, input_units

def category_to_name(probs, classes, cat_file):
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)
    cat_labels = [cat_to_name[i] for i in classes]
    return cat_labels
    
    
if __name__ == '__main__':
    main()
    