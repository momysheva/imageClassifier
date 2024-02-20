import torch
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
import json

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser (description = 'Parser _ prediction script')


parser.add_argument ('image_dir', help = 'Input image path. Mandatory', type = str)
parser.add_argument ('--load_dir', help = 'Checkpoint path. Optional', default = "checkpoint.pth", type = str)
parser.add_argument ('--top_k', help = 'Choose number of Top K classes. Default is 5', default = 5, type = int)
parser.add_argument ('--category_names', help = 'Provide path of JSON file mapping categories to names. Optional', default='cat_to_name.json', type = str)
parser.add_argument ('--gpu', type=str2bool, nargs='?', const=True, default=False, help="Use GPU or not")

args = parser.parse_args()

if args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'VGG-16-BN':
        model = models.vgg16_bn(weights=True)

    elif checkpoint['arch'] == 'VGG-13':
        model = models.vgg13(weights=True)
    
    elif checkpoint['arch'] == 'AlexNet':
        model = models.alexnet(weights=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    im = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    
    np_image = transform(im)

    return np_image

def predict(image_path, model, topk=5):

    model.to(device)

    img = process_image(image_path).unsqueeze_(0).float()
    img = img.to(device)

    model.eval()

    with torch.no_grad():
        out = model.forward(img)
        out = torch.exp(out)
    
    probs, classes = out.topk(topk, dim=1)
    

    probs = probs.numpy()[0]
    classes = classes.numpy()[0]
    

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in classes]
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    labels = [cat_to_name[i] for i in classes]

    return probs, labels

model = load_checkpoint(args.load_dir)

print("Image path: ", args.image_dir)
print("Predicting...")

probs, labels = predict(args.image_dir, model, args.top_k)

print("Top K classes: ", args.top_k)
for label, prob in zip(labels, probs):
    print(f"{label} : {prob*100:.2f}%")


