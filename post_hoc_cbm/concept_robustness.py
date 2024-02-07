import os
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import clip
import torch
import torch.nn as nn
from captum.attr import visualization as viz
import sys
import argparse
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm/models')
from models import get_derma_model
from data import load_ham_data
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')
sys.path.append('/data/gpfs/projects/punim2103/joint_training/finetuning_full_models') # Adding directory containing the model

# Concepts and classes
concept_list = concept_list = [                 ### change this line if working with HAM10k
    "air_conditioner", "basket", "blueness", "bumper", "ceramic", "countertop", "drinking_glass", 
    "fireplace", "greenness", "jar", "minibike", "pack", "plant", "water", "airplane", "bathroom_s", 
    "blurriness", "bus", "chain_wheel", "cow", "ear", "flag", "ground", "keyboard", "mirror", 
    "painted", "plate", "apron", "bathtub", "board", "bush", "chair", "cup", "earth", "floor", 
    "hair", "knob", "motorbike", "painting", "polka_dots", "arm", "beak", "body", "cabinet", 
    "chandelier", "curtain", "engine", "flower", "hand", "laminate", "mountain", "palm", "redness", 
    "armchair", "bed", "book", "can", "chest_of_drawers", "cushion", "exhaust_hood", "flowerpot", 
    "handle", "lamp", "mouse", "pane", "refrigerator", "ashcan", "bedclothes", "bookcase", 
    "candlestick", "chimney", "desk", "eye", "fluorescent", "handle_bar", "leather", "mouth", 
    "paper", "sand", "awning", "bedroom_s", "bottle", "canopy", "clock", "dining_room_s", "eyebrow", 
    "food", "head", "leg", "muzzle", "path", "snow", "back", "bench", "bowl", "cap", "coach", "dog", 
    "fabric", "foot", "headboard", "light", "neck", "paw", "sofa", "back_pillow", "bicycle", "box", 
    "car", "coffee_table", "door", "fan", "footboard", "headlight", "lid", "napkin", "pedestal", 
    "stairs", "bag", "bird", "brick", "cardboard", "column", "door_frame", "faucet", "frame", "hill", 
    "loudspeaker", "nose", "person", "street_s", "balcony", "blackness", "bridge", "carpet", 
    "computer", "doorframe", "fence", "glass", "horse", "manhole", "ottoman", "pillar", "stripes", 
    "bannister", "blind", "bucket", "cat", "concrete", "double_door", "field", "granite", "house", 
    "metal", "outside_arm", "pillow", "toilet", "base", "blotchy", "building", "ceiling", "counter", 
    "drawer", "figurine", "grass", "inside_arm", "microwave", "oven", "pipe", "tree"
]

idx_to_class = {                                ### change this line if working with HAM10k
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

# Sort the list alphabetically
sorted_concept_list = sorted(concept_list)

# Function to load adversarial images
def load_adversarial_images(batch_file):
    # Load the tensor from the .pt file
    adversarial_images = torch.load(batch_file, map_location=args.device)
    return adversarial_images

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder")
    parser.add_argument("--eps", default=0.001, required=True, type=float)    
    parser.add_argument("--norm", default='Linf', required=True, type=str)    
    parser.add_argument("--adv-dir", required=True, type=str, help="Path to the folder where the adversarial examples are located")  
    parser.add_argument("--clip_path", required=False, default="", type=int, help="Specify path if using the finetuned CLIP from experiment a.")
    parser.add_argument("--pcbm_path", required=False, default="", type=int, help="Specify path to the underlying pcbm or pcbm-h.")
    parser.add_argument("--finetuned_pcbm_path", required=False, default="", type=int, help="Specify path if the underlying pcbm has been finetuned.")
    parser.add_argument("--clip", required=False, default=False, type=bool, help="Set to True if finetuned-pcbm-path contains a jointly finetuned clip.")
    parser.add_argument("--batch-size", default=128, type=int)    
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=int, default=1)    
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()

# Load the model 
class ModelWrapper(nn.Module):
    def __init__(self, classifier, clip_model, resolution):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model
        
        # Define the preprocessing within the ModelWrapper
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        images = self.preprocess(images)
        print(images.shape)
        features = self.clip_model.encode_image(images)
        print(features.shape)
        out, x = self.classifier(features.float().to(args.device), return_dist = True)
        print(out.shape, x.shape)
        return out, x
    


args = config()  
save_dir = f"{args.out_dir}/concept_robustness_{args.norm}"
# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Load CLIP as backbone
model, _ = clip.load('RN50', args.device, jit=False)    ### change this line if working with HAM10k
resolution = 224
preprocess = transforms.Compose([                       ### change this line if working with HAM10k
        transforms.ToTensor(),
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

classifier = torch.load(args.pcbm_path, map_location=args.device) 

if args.clip_path != "": # update clip if the finetuned version from experiment a was used
    checkpoint = torch.load(args.clip_path, map_location = args.device)
    clip_model_state_dict = checkpoint['clip_model_state_dict']   
    model.load_state_dict(clip_model_state_dict)

if args.finetuned_pcbm_path != "": # update pcbm if it was gefinetuned
    checkpoint = torch.load(args.finetuned_pcbm_path, map_location = args.device)
    if args.clip == True:
        clip_model_state_dict = checkpoint['clip_model_state_dict']   
        model.load_state_dict(clip_model_state_dict)
    classifier_state_dict = checkpoint['pcbm_model_state_dict']
    classifier.load_state_dict(classifier_state_dict)

wrapped_model = ModelWrapper(classifier, model, resolution).to(args.device)
wrapped_model.eval()

# Load CIFAR-10 testset
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(1000))
test_loader = DataLoader(test_subset, batch_size=args.batch_size)

# Load the adversarial images
adversarial_dir = args.adv_dir
adversarial_files = sorted(os.listdir(adversarial_dir))

# Function to compare concepts
def compare_concepts(original_concepts, adversarial_concepts, top_n):
    return len(set(original_concepts[:top_n]).intersection(adversarial_concepts[:top_n])) / top_n * 100

results = []
batch = 0

# Iterate over both the original and adversarial images simultaneously
for i, (data, filename) in enumerate(zip(test_loader, adversarial_files)):
    print(batch)
    orig_images, labels = data
    orig_images, labels = orig_images.to(args.device), labels.to(args.device)

    # Load adversarial images for the current batch
    batch_filename = f"eps_{args.eps}_batch_{i+1}_adv.pt"  # Construct the filename
    print(batch_filename)
    batch_file = os.path.join(adversarial_dir, batch_filename)
    adversarial_images = load_adversarial_images(batch_file).to(args.device)

    # Make sure the shapes match, otherwise, there is a batch size mismatch
    assert adversarial_images.shape[0] == orig_images.shape[0], "Batch sizes do not match."

    # Get the model outputs for the original images
    orig_outputs, orig_dist = wrapped_model(orig_images)
    _, orig_predictions = torch.max(orig_outputs, 1)

    # Get the model outputs for the adversarial images
    adv_outputs, adv_dist = wrapped_model(adversarial_images)
    _, adv_predictions = torch.max(adv_outputs, 1)

    # multiply distances with weights to get contributions 
    weights = wrapped_model.classifier.bottleneck.classifier.weight
    bias = wrapped_model.classifier.bottleneck.classifier.bias

    # Initialize contributions tensor
    orig_contributions = torch.zeros_like(orig_dist)
    adv_contributions = torch.zeros_like(adv_dist)

    # Calculate contributions for each image in the batch
    for k in range(orig_dist.size(0)):  # Loop over the batch
        orig_class_idx = orig_predictions[k]  # Get predicted class index for each image
        adv_class_idx = adv_predictions[k]  # Get predicted class index for each image

        orig_class_weights = weights[orig_class_idx]  # Get weights for the predicted class
        adv_class_weights = weights[adv_class_idx]  # Get weights for the predicted class
        # as in saliency_maps we can delete these two lines as the bias if we are looking only at the top class
        orig_class_bias = bias[orig_class_idx]  # Get bias for the predicted class
        adv_class_bias = bias[adv_class_idx]  # Get bias for the predicted class

        orig_contributions[k] = orig_dist[k] * orig_class_weights + orig_class_bias
        adv_contributions[k] = adv_dist[k] * adv_class_weights + adv_class_bias


    # Get the top 50 influential concepts for original images
    _, orig_influential_indices = torch.topk(abs(orig_contributions), 50, largest=True) # return largest 
    orig_concepts = [[sorted_concept_list[idx] for idx in batch] for batch in orig_influential_indices]

    # Get the top 50 influential concepts for adversarial images
    _, adv_influential_indices = torch.topk(abs(adv_contributions), 50, largest=True)
    adv_concepts = [[sorted_concept_list[idx] for idx in batch] for batch in adv_influential_indices]

    # Compare concepts and store results including correct and predicted labels
    for j in range(adversarial_images.size(0)):
        correct_label = idx_to_class[labels[j].item()]
        predicted_label_orig = idx_to_class[orig_predictions[j].item()]
        predicted_label_adv = idx_to_class[adv_predictions[j].item()]

        # Compute L2 and Linf norm between concept vectors
        l2_distance = torch.norm(orig_dist[j] - adv_dist[j], p=2)
        linf_distance = torch.norm(orig_dist[j] - adv_dist[j], p=float('inf'))

        # Initialize a dictionary to store the results for this image
        result = {
            'Image Index': i * args.batch_size + j,
            'Correct': correct_label,
            'Predicted Orig': predicted_label_orig,
            'Predicted Adv': predicted_label_adv,
            'L2 Distance': l2_distance.item(),
            'Linf Distance': linf_distance.item(),
            'Top-5': None,
            'Top-10': None,
            'Top-15': None,
            'Top-20': None,
            'Top-50': None
        }

        for top_n in [5, 10, 15, 20, 50]:
            percent = compare_concepts(orig_concepts[j], adv_concepts[j], top_n)
            result[f'Top-{top_n}'] = percent

        results.append(result)

    batch += 1
    #if batch == 21:
        #break

# Convert results to a DataFrame and save as CSV
results_df = pd.DataFrame(results, columns=[
    'Image Index', 'Correct', 'Predicted Orig', 'Predicted Adv',
    'L2 Distance', 'Linf Distance', 'Top-5', 'Top-10', 'Top-15', 'Top-20', 'Top-50'
])
results_df.to_csv(f"{save_dir}/concept_robustness_cifar_{args.eps}.csv", index=False)
