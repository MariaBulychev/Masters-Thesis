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
import plots
import torchvision.utils as vutils
import os
import argparse
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')

concept_list = [            # use this list if working with the broden concept bank 
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
    parser.add_argument("--seed", default=42, type=int, help="Random seed")    
    parser.add_argument("--batch-size", default=128, type=int)    
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=int, default=1)    
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()

# Sort the list alphabetically
sorted_concept_list = sorted(concept_list)

idx_to_class = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

def visualize_image_attr(
    attribution,
    method="heat_map",
    sign="absolute_value",
    plt_fig_axis=None,
    outlier_perc=2,
):
    if plt_fig_axis is None:
        fig, axis = plt.subplots(figsize=(6, 6))
    else:
        fig, axis = plt_fig_axis

    # Process attribution for visualisation
    attr = attribution.squeeze().cpu().detach().numpy()
    
    # Ensure the attribution is two-dimensional
    if attr.ndim == 3:
        attr = np.transpose(attr, (1, 2, 0))
        attr = np.mean(attr, axis=2)
    elif attr.ndim == 1:  # If the saliency map is one-dimensional
        attr = np.expand_dims(attr, axis=0)  # Make it two-dimensional

    if sign == "absolute_value":
        attr = np.abs(attr)
    elif sign == "positive":
        attr = np.clip(attr, 0, 1)
    elif sign == "negative":
        attr = -np.clip(attr, -1, 0)

    if method == "heat_map":
        vmin, vmax = np.percentile(attr, [outlier_perc, 100 - outlier_perc])
        axis.imshow(attr, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        raise NotImplementedError("Visualisation method not implemented.")

    axis.axis('off')
    axis.set_title('Attribution Map')

    plt.tight_layout()
    return fig, axis

def get_saliency_maps_grid(image, i, label, wrapped_model, values, feature_names, max_display, save_path, adv=False):
    contributions = values
    image, label = image.to(args.device), label.to(args.device)
    outputs, dist = wrapped_model(image)  

    if isinstance(contributions, np.ndarray):
        contributions = torch.from_numpy(contributions)
    
    _, predicted = torch.max(outputs, 1)
    predicted_label = [idx_to_class[predicted.item()]]
    true_label = [idx_to_class[label.item()]]

    _, top_contrib_indices = torch.topk(abs(contributions), k=7)
    top_concept_names = [feature_names[idx] for idx in top_contrib_indices.cpu().numpy()]

    wrapped_model.return_dist_only = True
    ig = IntegratedGradients(wrapped_model)

    
    baselines = ['black', 'white']

    # Generate saliency maps for each concept and add them to the grid
    for j in range(len(baselines)): # for black and white baseline
        # Create a grid for subplots: 2 rows, 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        # Convert the original image to numpy format for visualization
        np_img = np.transpose(image.cpu().squeeze(0).numpy(), (1, 2, 0))

        # Visualize original image in the first subplot
        axes[0, 0].imshow(np_img)
        axes[0, 0].axis('off')
        axes[0, 0].set_title(f'Original Image\nTrue: {true_label[0]}, Pred: {predicted_label[0]}')

        for idx, concept_name in enumerate(top_concept_names):
            
            row, col = divmod(idx + 1, 4)
            concept_idx = feature_names.index(concept_name)

            # Calculate the attribution for the current concept
            attributions = ig.attribute(image, baselines=j, target=int(concept_idx), return_convergence_delta=False, method='gausslegendre')

            # Visualize the attribution image using your function
            visualize_image_attr(
                attribution=attributions,
                plt_fig_axis=(fig, axes[row, col])
            )
            
            # Adjust subplot titles or any other customization
            axes[row, col].set_title(f'Concept: {concept_name}')

        plt.tight_layout()
        if adv == False:
            plt.savefig(f"{save_path}/combined_attribution_maps_{baselines[j]}_baseline_image_{i}.png")
        else:
            plt.savefig(f"{save_path}/combined_attribution_maps_{baselines[j]}_baseline_image_{i}_adv.png")
        plt.clf()

class ModelWrapper(nn.Module):
    def __init__(self, classifier, clip_model, resolution, return_dist_only = False):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model
        self.return_dist_only = return_dist_only
        
        # Define the preprocessing pipeline within the ModelWrapper
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        images = self.preprocess(images)
        features = self.clip_model.encode_image(images)
        out, x = self.classifier(features.float().to(args.device), return_dist = True)
        if self.return_dist_only:
            return x 
        else:
            return out, x
        
def load_adversarial_images(batch_file):
    # Load the tensor from the .pt file
    adversarial_images = torch.load(batch_file, map_location=args.device)
    return adversarial_images
    
    
args = config()  
save_dir = f"{args.out_dir}/saliency_maps_{args.norm}"
# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Load finetuned CLIP as backbone
model, _ = clip.load('RN50', args.device, jit=False)
resolution = 224
preprocess = transforms.Compose([
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

test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
#test_subset = Subset(test_dataset, range(128)) # loading a subset of cifar
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Load the adversarial images
adversarial_dir = args.adv_dir
adversarial_files = sorted(os.listdir(adversarial_dir))

# Initialise batch counter
batch_number = 1

# DataLoader setup
dataiter = iter(test_loader)
images, labels = next(dataiter)

classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

found_img = 0

with torch.no_grad():
    i = 126
    while True:
        i += 1
        if i >= 128:
            print("Next batch")
            batch_number += 1  # Increase batch number
            i=0
            batch_filename = f"eps_{args.eps}_batch_{batch_number}_adv.pt"
            batch_file = os.path.join(adversarial_dir, batch_filename)
            adversarial_images = load_adversarial_images(batch_file).to(args.device)

            # Try to load the next batch of data, break if no more data
            try:
                images, labels = next(dataiter)
            except StopIteration:
                print("No more data")
                break  # No more data, exit the loop

        wrapped_model.return_dist_only = False

        image = images[i].unsqueeze(0).to(args.device)
        label = labels[i].to(args.device)

        adv_image = adversarial_images[i].unsqueeze(0).to(args.device)

        orig_pred, _ = wrapped_model(image)
        _, orig_predictions = torch.max(orig_pred, 1)
        adv_pred, _ = wrapped_model(adv_image)
        _, adv_predictions = torch.max(adv_pred, 1)

        if orig_predictions != adv_predictions:

            found_img +=1
            if found_img >=11:
                break

            print(f"Batch {i}: Original image classified as {orig_predictions}, adversarial image classified as {adv_predictions}")

            folder_name = os.path.join(save_dir, f"batch_{batch_number}_image_{i}")
            print(f"Folder name: {folder_name}")
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Save the image
            image_path = os.path.join(folder_name, f"image_{i}.png")
            vutils.save_image(image, image_path)
            
            wrapped_model.return_dist_only = False
            outputs, concept_act = wrapped_model(image)
            
            top_logit_vals, top_classes = torch.topk(outputs[0], dim=0, k=2)
            conf = torch.nn.functional.softmax(outputs[0], dim=0)

            print("Image:{} Gt:{}, 1st Pred:{}, {:.3f}, 2nd Pred:{}, {:.3f}".format(i, classes[int(label)], classes[top_classes[0]], top_logit_vals[0],
                                                                        classes[top_classes[1]], top_logit_vals[1]))
            
            for k in range(1):
                weights = wrapped_model.classifier.bottleneck.classifier.weight
                bias = wrapped_model.classifier.bottleneck.classifier.bias
                contributions = concept_act[0]*weights[top_classes[k], :] # concept_act = distances

                #######################################################################################################
                _, top_contrib_indices = torch.topk(abs(contributions), k=5)
                top_contributions = contributions[top_contrib_indices]
                top_activations = concept_act[0][top_contrib_indices]

                # Retrieve the names of the top concepts
                top_concept_names = [sorted_concept_list[idx] for idx in top_contrib_indices.cpu().numpy()]

                # Print top 5 contributions, corresponding concept activations, and concept names
                print(f"Top 5 contributions for class '{classes[top_classes[k]]}': {top_contributions}")
                print(f"Corresponding activations: {top_activations}")
                print(f"Top concept names: {top_concept_names}")

                #######################################################################################################

                feature_names = [("NOT " if concept_act[0][i] < 0 else "") + sorted_concept_list[i] for i in range(len(sorted_concept_list))]
                values = contributions.cpu().numpy()
                #max_display = min(int(sum(abs(values)>0.005))+1, 8)
                max_display = 7
                title = "Pred:{} - Conf: {:.3f} - Logit:{:.2f} - Bias:{:.2f}".format(classes[top_classes[k]],
                                conf[top_classes[k]], top_logit_vals[k], wrapped_model.classifier.bottleneck.classifier.bias[top_classes[k]])
                attributions_path = os.path.join(folder_name, f"attributions_{i}_orig.png")
                plots.bar(values, feature_names, max_display=max_display, title=title, fontsize=16, save_path=attributions_path)

                get_saliency_maps_grid(image, i, label, wrapped_model, values, feature_names, max_display=max_display, save_path=folder_name)

            ################ repeat everything for adversarial image 

            # Save the image
            adv_image_path = os.path.join(folder_name, f"adv_image_{i}.png")
            vutils.save_image(adv_image, adv_image_path)
            
            wrapped_model.return_dist_only = False
            adv_outputs, adv_concept_act = wrapped_model(adv_image)
            
            adv_top_logit_vals, adv_top_classes = torch.topk(adv_outputs[0], dim=0, k=2)
            adv_conf = torch.nn.functional.softmax(adv_outputs[0], dim=0)

            print("Image:{} Gt:{}, 1st Pred:{}, {:.3f}, 2nd Pred:{}, {:.3f}".format(i, classes[int(label)], classes[adv_top_classes[0]], adv_top_logit_vals[0],
                                                                        classes[adv_top_classes[1]], adv_top_logit_vals[1]))
            
            for k in range(1):
                weights = wrapped_model.classifier.bottleneck.classifier.weight
                bias = wrapped_model.classifier.bottleneck.classifier.bias
                adv_contributions = adv_concept_act[0]*weights[adv_top_classes[k], :] # concept_act = distances

                #######################################################################################################
                _, adv_top_contrib_indices = torch.topk(abs(adv_contributions), k=5)
                adv_top_contributions = adv_contributions[adv_top_contrib_indices]
                adv_top_activations = adv_concept_act[0][adv_top_contrib_indices]

                # Retrieve the names of the top concepts
                adv_top_concept_names = [sorted_concept_list[idx] for idx in adv_top_contrib_indices.cpu().numpy()]

                # Print top 5 contributions, corresponding concept activations, and concept names
                print(f"Top 5 contributions for class '{classes[adv_top_classes[k]]}': {adv_top_contributions}")
                print(f"Corresponding activations: {adv_top_activations}")
                print(f"Top concept names: {adv_top_concept_names}")
                #######################################################################################################

                adv_feature_names = [("NOT " if adv_concept_act[0][i] < 0 else "") + sorted_concept_list[i] for i in range(len(sorted_concept_list))]
                adv_values = adv_contributions.cpu().numpy()
                max_display = 7
                title = "Pred:{} - Conf: {:.3f} - Logit:{:.2f} - Bias:{:.2f}".format(classes[adv_top_classes[k]],
                                adv_conf[adv_top_classes[k]], adv_top_logit_vals[k], wrapped_model.classifier.bottleneck.classifier.bias[adv_top_classes[k]])
                adv_attributions_path = os.path.join(folder_name, f"attributions_{i}_adv.png")
                plots.bar(adv_values, adv_feature_names, max_display=max_display, title=title, fontsize=16, save_path=adv_attributions_path)

                get_saliency_maps_grid(adv_image, i, label, wrapped_model, adv_values, adv_feature_names, max_display=max_display, save_path=folder_name, adv=True)
