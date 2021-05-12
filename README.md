# Flower Classification (102 different flower categories)
Flower Image Classification using the Oxford 102 Flower Dataset and Transfer Learning Techniques. 

## Get Data
Run setup.sh to download the Oxford 102 Flower Dataset. 

##  Training the Model
To train the model, run train.py with the correct arguments to train the model on the provided dataset.

#### Example Call

```
python train.py --data_dir ../aipnd-project/flowers --epochs 4 --arch vgg19 --learning_rate 0.0003 --gpu 0 --store_dir data --hidden_units 4096
```

### Usage
```
python train.py [-h] --data_dir DATA_DIR [--arch ARCH] [--save_dir SAVE_DIR]
                [--learning_rate LEARNING_RATE] [--epochs EPOCHS] [--gpu GPU]
                [--hidden_units HIDDEN_UNITS [HIDDEN_UNITS ...]]

DATA_DIR      = Path to flower directory 
ARCH          = Pretrained Network Architecture (Available options VGG , Densenet & Resnet)
SAVE_DIR      = Location to save model 
LEARNING RATE = Learning Rate 
EPOCHS        = Number of epochs to train for
GPU           = Train with GPU (0 is for GPU on, negative number for CPU)
HIDDEN UNITS  = Hidden units list
```

Run ```python train.py -h``` to see the help documentation for the function

Training may take some timne, depending on the amount of compute resources available 

##  Making a prediction 
To make a prediction, run predict.py with the desired checkpoint file and the location of the image.


#### Example Call

```
python predict.py -p data/checkpoint.pth -f 'aipnd-project/flowers/test/29/image_04083.jpg' -topk 4 -g 0 -c 'cat_to_name.json
```

### Usage
```
python predict.py [-h] --chk_path CHK_PATH [--imgpath IMGPATH] [--topk TOPK]
                  [--gpu GPU] [--cat_file CAT_FILE]

CHK_PATH      = Path to checkpoint file 
IMGPATH       = Path to Image to be tested 
TOPK          = Top K predictions of the model for the input image
GPU           = Train with GPU (0 is for GPU on, negative number for CPU)
CAT_FILE      = Path to category-to-name mapping file
```

Run ```python predict.py -h``` to see the help documentation for the function
