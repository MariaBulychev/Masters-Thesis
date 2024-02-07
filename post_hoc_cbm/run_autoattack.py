import sys
import clip
import torch
import torch.nn as nn
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')
from autoattack import AutoAttack
from torchvision.transforms import ToPILImage
import argparse
import os

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder to save the adversarial examples.")
    parser.add_argument("--eps", default=0.001, required=True, type=float)
    parser.add_argument("--norm", default="Linf", required=True, type=str)
    parser.add_argument("--clip_path", required=False, default="", type=int, help="Specify path if using the finetuned CLIP from experiment a.")
    parser.add_argument("--pcbm_path", required=False, default="", type=int, help="Specify path to the underlying pcbm or pcbm-h.")
    parser.add_argument("--finetuned_pcbm_path", required=False, default="", type=int, help="Specify path if the underlying pcbm has been finetuned.")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


class ModelWrapper(nn.Module):
    def __init__(self, classifier, clip_model, resolution):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        images = self.preprocess(images)
        features = self.clip_model.encode_image(images)
        outputs = self.classifier(features.float().to(args.device))
        return outputs

args = config()
save_dir = f"{args.out_dir}/model_robustness_{args.norm}"
# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Load CLIP as backbone
model, _ = clip.load('RN50', args.device, jit=False)        ### change this line if working with HAM10k
resolution = 224
preprocess = transforms.Compose([                           ### change this line if working with HAM10k
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

print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(1000)) # to evaluate on the first 1000 images
test_loader = DataLoader(test_subset, batch_size=args.batch_size)

epsilon = args.eps

adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard', device=args.device)

csv_path = f'{save_dir}/robustness_eps_{epsilon}.csv'
batch = 0
results = []

with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Epsilon", "Initial Accuracy", "Robust Accuracy", "Max Perturbation"])  # Header

    for images, labels in test_loader:
        batch += 1        
        images, labels = images.to(args.device), labels.to(args.device)

        outputs = wrapped_model(images)
        _, predicted = torch.max(outputs, 1)
        initial_acc = (predicted == labels).sum().item() / images.shape[0]
        print(f'Initial Accuracy for Batch {batch}: {100 * initial_acc:.2f}%')

        x_adv, robust_accuracy, res = adversary.run_standard_evaluation(images, labels, bs=images.shape[0])

        torch.save(x_adv, f'{save_dir}/eps_{epsilon}_batch_{batch}_adv.pt')

        results.append([100 * initial_acc, 100* robust_accuracy, res.item()])
        csv_writer.writerow([epsilon, 100 * initial_acc, 100 * robust_accuracy, res.item()])
        
    # Calculate and write the mean values at the end of the csv
    mean_values = [epsilon] + [sum(col)/len(col) for col in zip(*results)]
    csv_writer.writerow([])
    csv_writer.writerow(["Mean for Epsilon", "Mean Initial Accuracy", "Mean Robust Accuracy", "Mean Max Perturbation"])
    csv_writer.writerow(mean_values)
