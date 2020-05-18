import prediction_arguments
import json
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms



def main():
    """
    Predictions are happening here!
    """
    
    
    
    
    print("predict!")
    parser = prediction_arguments.prediction_args()
    args = parser.parse_args()
    
    #CPU OR GPU?
    device = torch.device("cpu")
    if args.gpu:
        device = torch.device("cuda")
        
    
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_the_model(device, args.checkpoint)
    
    top_probs, actual_names = predict(args.path_to_image, model, args.top_k, cat_to_name, device)
    
    print("-------------------------------------------")
    print("Current image: {}".format(args.path_to_image))
    print("Model: {}".format(args.checkpoint))
    print("Device: {}".format(device))
    print("-------------------------------------------")
    print("Our predictions:")
    print("The flower is {} with probability {:.2f}".format(actual_names[0], top_probs[0]*100))
    print("-------------------------------------------")
    print("Top possibilities are: ")
    for i in range(len(top_probs)):
        print("the flower is {} with probability {:.2f}".format(actual_names[i], top_probs[i]*100))
        
    
    
    
    
    
def load_the_model(device, checkpoint_name):
    
    dict_model = torch.load(checkpoint_name)
    model = models.__dict__[dict_model["arch"]](pretrained=True)
    model.classifier = dict_model["classifier"]
    model.class_to_idx = dict_model["class_to_idx"]
    model.load_state_dict(dict_model["state_dict"])
    model.to(device)

    
   

    return model

    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    min_dimension = 0 
    
    if image.size[1] < image.size[0]:
        min_dimension = 1
    
    if min_dimension == 0:
        image.thumbnail((256, image.size[1]))
    else:
        image.thumbnail((image.size[0], 256))
        
    ##### come of these transforms can and should be done via transforms.Compose() 
    
    ########### crop   
    box = ((image.width-224)/2, (image.height-224)/2, (image.width-224)/2 + 224, (image.height-224)/2 + 224 )
    image = image.crop(box = box)

    ########### convert
    image = np.array(image)/255     
    image = (image - np.array([0.485, 0.456, 0.406]) )/np.array([0.229, 0.224, 0.225])
    
    
    #########transpose, hope it does what it's supposed to do
    image = image.transpose( (2, 0, 1) )


    return image


def predict(image_path, model, topk, cat_to_name, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    im = Image.open(image_path)
    numpy_im = process_image(im)
    
    
    
    if device.type == "cuda":
        images = torch.from_numpy(numpy_im).type(torch.cuda.FloatTensor)
    else:
        images = torch.from_numpy(numpy_im).type(torch.FloatTensor)
    
#     print(images.shape)
   
    #############batch size???##########
    images.unsqueeze_(0)
#     print(images.shape)
    
    
    with torch.no_grad():
        model.eval()
        log_probs = model(images)
    probs = torch.exp(log_probs)
    top_probs, top_classes = probs.topk(topk)

    
    
    idx_into_class = {y: x for x, y in model.class_to_idx.items()}

    ########into lists
    top_probs = top_probs.cpu().numpy()
    top_probs = top_probs.tolist()[0]
    
    top_classes = top_classes.cpu().numpy()
    top_classes = top_classes.tolist()[0]
    
    
    
    
    actual_names = [cat_to_name[idx_into_class[model_out_class]] for model_out_class in top_classes]
     
    
    return top_probs, actual_names
    
    
    
  
if __name__== "__main__":
    main()


