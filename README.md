# Machine learning image classifier using PyTorch
This is an image classifier for this dataset: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. It runs from a command line using argparse and pytorch. 

The first file, train.py, trains a classifier (densenet or vgg) and saves the trained model as a checkpoint. The second file, predict.py, uses the saved model to predict the class for an input image. 

# Examples for the train.py:

Help:
python train.py -h

Basic usage:
'''
python train.py ./flowers/
'''
Prints out training loss, validation loss, and validation accuracy as the network trains.

Checkpoint saved to model1 directory :
python train.py data_dir --save_dir model1

Choose architecture: 
python train.py data_dir --arch vgg11

Set parameters: 
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 10

Enable CUDA:
python train.py data_dir --gpu


# Examples for the predict.py:

Help:
python predict.py -h

Prediction:
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth

Outputs top 7 probabilities:
python predict.py input checkpoint --top_k 7

Uses CUDA for the forward step: 
python predict.py input checkpoint --gpu

