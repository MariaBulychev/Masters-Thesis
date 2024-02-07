import sys
import os
import torch
from collections import defaultdict
import numpy as np
from torchvision import datasets, transforms
import subprocess
import sklearn
from sklearn.svm import SVC
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn

class SVMMimic(nn.Module):
    def __init__(self, in_features, out_features, weights, intercept):
        super(SVMMimic, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        
        # Set the weights and intercept from the SVM
        self.fc.weight = nn.Parameter(torch.from_numpy(weights).float())
        self.fc.bias = nn.Parameter(torch.from_numpy(intercept).float())

    def forward(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)
        x = self.fc(x)
        return x


class ListDataset:
    def __init__(self, images, preprocess=None):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.preprocess:
            image = self.preprocess(image)
        return image


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(
                val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]


@torch.no_grad()
def get_embeddings(loader, model, device="cuda"):
    """
    Args:
        loader ([torch.utils.data.DataLoader]): Data loader returning only the images
        model ([nn.Module]): Backbone
        n_samples (int, optional): Number of samples to extract the activations
        device (str, optional): Device to use. Defaults to "cuda".

    Returns:
        np.array: Activations as a numpy array.
    """
    activations = None
    for image in tqdm(loader):
        image = image.to(device)
        print(image.shape)
        try:
            batch_act = model(image).squeeze().detach().cpu().numpy()
        except:
            # Then it's a CLIP model. This is a really nasty soln, i should fix this.
            # preprocess here weil in data entfernt 
            preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
            image = preprocess(image)
            batch_act = model.encode_image(image).squeeze().detach().cpu().numpy()
        if activations is None:
            activations = batch_act
        else:
            activations = np.concatenate([activations, batch_act], axis=0)
    return activations


def get_cavs(X_train, y_train, X_val, y_val, C):
    """Extract the concept activation vectors and the corresponding stats

    Args:
        X_train, y_train, X_val, y_val: activations (numpy arrays) to learn the concepts with.
        C: Regularizer for the SVM. 
    """
    svm = SVC(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_val, y_val)
    train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
    margin_info = {"max": np.max(train_margin),
                   "min": np.min(train_margin),
                   "pos_mean":  np.nanmean(train_margin[train_margin > 0]),
                   "pos_std": np.nanstd(train_margin[train_margin > 0]),
                   "neg_mean": np.nanmean(train_margin[train_margin < 0]),
                   "neg_std": np.nanstd(train_margin[train_margin < 0]),
                   "q_90": np.quantile(train_margin, 0.9),
                   "q_10": np.quantile(train_margin, 0.1),
                   "pos_count": y_train.sum(),
                   "neg_count": (1-y_train).sum(),
                   }
    concept_info = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)
    return concept_info

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, criterion, X, target, alpha, attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    upper_limit, lower_limit = 1, 0
    delta = torch.zeros_like(X).cuda()

    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)

    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon

    else:
        raise ValueError
    
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    for _ in range(attack_iters):
        # output = model(normalize(X ))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        output = model(X+delta)
        output = output.view_as(target).to(device)
        target = target.to(device)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)

        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta

class ModelWrapper(nn.Module):
    def __init__(self, clip_model, svm):
        super().__init__()
        self.clip_model = clip_model
        self.svm = svm

        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        images = self.preprocess(images)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = images.to(device)
        image_features = self.clip_model.encode_image(images)
        return self.svm(image_features.float())
    
def get_cavs_adv(X_train, y_train, X_val, y_val, C, pos_loader, neg_loader, clip_model):    
    """Extract the concept activation vectors and the corresponding stats

    Args:
        X_train, y_train, X_val, y_val: activations (numpy arrays) to learn the concepts with.
        C: Regularizer for the SVM. 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    svm = SVC(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    # mimic the svm with a neural network
    weights = svm.coef_
    intercept = svm.intercept_
    n_features = X_train.shape[1]
    svm_model = SVMMimic(in_features=n_features, out_features=1, weights=weights, intercept=intercept).to(device)
    wrapped_model = ModelWrapper(clip_model, svm_model).to(device)
    preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    # generate adversarial examples
    for images in tqdm(pos_loader): # gehe durch die originalen positiven bilder  nicht preprocessed
        images = images.to(device)
        print(images.shape)
        criterion = nn.BCEWithLogitsLoss()
        labels = torch.ones(1,images.shape[0])
        print(labels)
        delta = attack_pgd(wrapped_model, criterion, images, labels, alpha=1/255, attack_iters=5, norm='l_inf', restarts=1, early_stop=True, epsilon=0.001)       
        adv_images = images + delta
        #preprocess adv images 
        adv_images = preprocess(adv_images)
        adv_activations = clip_model.encode_image(adv_images.float()).squeeze().detach().cpu().numpy()

        X_train = np.concatenate((X_train, adv_activations), axis=0)
        new_labels = np.ones(adv_images.shape[0])
        y_train = np.concatenate((y_train, new_labels), axis=0)

        
    for images in tqdm(neg_loader): # gehe durch die originalen negativen bilder  nicht preprocessed
        images = images.to(device)
        print(images.shape)
        criterion = nn.BCEWithLogitsLoss()
        labels = torch.zeros(1,images.shape[0])
        print(labels)
        delta = attack_pgd(wrapped_model, criterion, images, labels, alpha=1/255, attack_iters=5, norm='l_inf', restarts=1, early_stop=True, epsilon=0.001)       
        adv_images = images + delta
        #preprocess adv images 
        adv_images = preprocess(adv_images)

        adv_activations = clip_model.encode_image(adv_images.float()).squeeze().detach().cpu().numpy()
        X_train = np.concatenate((X_train, adv_activations), axis=0)
        y_train = np.concatenate((y_train, np.zeros(adv_images.shape[0])), axis=0)

    # fit SVM once again
    svm.fit(X_train, y_train)

    # rest 
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_val, y_val)
    train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
    margin_info = {"max": np.max(train_margin),
                   "min": np.min(train_margin),
                   "pos_mean":  np.nanmean(train_margin[train_margin > 0]),
                   "pos_std": np.nanstd(train_margin[train_margin > 0]),
                   "neg_mean": np.nanmean(train_margin[train_margin < 0]),
                   "neg_std": np.nanstd(train_margin[train_margin < 0]),
                   "q_90": np.quantile(train_margin, 0.9),
                   "q_10": np.quantile(train_margin, 0.1),
                   "pos_count": y_train.sum(),
                   "neg_count": (1-y_train).sum(),
                   }
    concept_info = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)
    return concept_info

def learn_concept_bank(pos_loader, neg_loader, backbone, n_samples, C, device="cuda", adv=False):
    """Learning CAVs and related margin stats.
    Args:
        pos_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding positive samples for each concept
        neg_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding negative samples for each concept
        model_bottom (nn.Module): Mode
        n_samples (int): Number of positive samples to use while learning the concept.
        C (float): Regularization parameter for the SVM. Possibly multiple options.
        device (str, optional): Device to use while extracting activations. Defaults to "cuda".

    Returns:
        dict: Concept information, including the CAV and margin stats.
    """

    print("Extracting Embeddings: ")
    pos_act = get_embeddings(pos_loader, backbone, device=device)
    neg_act = get_embeddings(neg_loader, backbone, device=device)
    
    X_train = np.concatenate([pos_act[:n_samples], neg_act[:n_samples]], axis=0)
    X_val = np.concatenate([pos_act[n_samples:], neg_act[n_samples:]], axis=0)
    y_train = np.concatenate([np.ones(pos_act[:n_samples].shape[0]), np.zeros(neg_act[:n_samples].shape[0])], axis=0)
    y_val = np.concatenate([np.ones(pos_act[n_samples:].shape[0]), np.zeros(neg_act[n_samples:].shape[0])], axis=0)
    concept_info = {}
    for c in C:
        # If concepts are learned employing adversarial examples (experiments h and i)
        if adv == True: # Experiments i & h 
            concept_info[c] = get_cavs_adv(X_train, y_train, X_val, y_val, c, pos_loader, neg_loader, backbone)
        else: # All remaining experiments
            concept_info[c] = get_cavs(X_train, y_train, X_val, y_val, c, pos_loader, neg_loader, backbone)
    return concept_info
