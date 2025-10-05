from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from torchvision.models import resnet18, resnet50
import pandas as pd
import torch
import numpy as np
import random
from torch.optim import Adam
import os
from torchvision.datasets import Omniglot
from tqdm import tqdm
# from easyfsl.data_tools import TaskSampler
from torch import nn, optim
from torchvision.datasets import ImageFolder
from easyfsl.utils import plot_images, sliding_average
from torchvision.models import resnet18, resnet50
import timm
from model_zoo import PrototypicalNetworks, Trainer
from samplers import TaskSampler, ImageFolderWithPaths
from shap_image import PrototypicalSHAPExplainer
from datetime import datetime

if __name__ == "__main__":
    seed = 433
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Parameters ---
    image_size = 384
    N_WAY = 2
    N_SHOT = 3 # sample per class
    N_QUERY = 3
    N_TRAINING_EPISODES = 500 # Train
    N_EVALUATION_TASKS = 100 # Test
    N_VALIDATION_EPISODES = 20 # Val
    log_update_frequency = 50
    val_interval = 20

    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        # transforms.Resize([int(image_size), int(image_size)]),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize([int(image_size), int(image_size)]),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    # --- Datasets ---
    base_path = "/Users/buddy/Documents/Vscode/opthalfsl/src2/dataset"
    train_set = ImageFolderWithPaths(os.path.join(base_path, "train"), transform=train_transform)
    train_set.labels = train_set.targets
    train_set.get_labels = lambda: train_set.targets  # <- ADD THIS

    val_set = ImageFolderWithPaths(os.path.join(base_path, "val"), transform=val_transform)
    val_set.labels = val_set.targets
    val_set.get_labels = lambda: val_set.targets      # <- ADD THIS

    # --- Samplers and Loaders ---
    train_sampler = TaskSampler(train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=0,           # or more if Colab Pro
        pin_memory=True,         # helps host→GPU speed
        collate_fn=train_sampler.episodic_collate_fn,
        # persistent_workers=True, # if using PyTorch 1.7+
        # prefetch_factor=4,
    )

    val_sampler = TaskSampler(val_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_VALIDATION_EPISODES)
    val_loader = DataLoader(
        val_set, 
        batch_sampler=val_sampler, 
        num_workers=0,
        pin_memory=False, 
        collate_fn=val_sampler.episodic_collate_fn)

    backbone_name = "vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k"
    # Load backbone
    # cnn_backbone = resnet18(pretrained=True) # 512
    # cnn_backbone = resnet50(pretrained=True) # 2048
    # cnn_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) # 1000
    # cnn_backbone = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in12k', pretrained=True, num_classes=0) # 768
    # cnn_backbone = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0) # 2048
    # cnn_backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0) # 1280
    cnn_backbone = timm.create_model('timm/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', pretrained=True, num_classes=0) # 512 # 384
    # cnn_backbone = timm.create_model('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384', pretrained=True, num_classes=0) # 1536 # 256 images size

    cnn_backbone.fc = nn.Identity()

    for param in cnn_backbone.parameters():
        param.requires_grad = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noted = 'center_crop'

    model = PrototypicalNetworks(cnn_backbone).to(device)
    model_path = f"models/best_{backbone_name}_{N_TRAINING_EPISODES}_{image_size}_{timestamp}_{noted}.pth"
    log_file = f"logs/training_log_{backbone_name}_{N_TRAINING_EPISODES}_{image_size}_{timestamp}_{noted}.csv"
    result_file = f"results/evaluation_results_{backbone_name}_{N_TRAINING_EPISODES}_{image_size}_{timestamp}_{noted}.csv"

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=0.0001,
        patience=10,
        model_path=model_path,  # Saved best model
        log_file=log_file,  # Logs training stats
        result_file=result_file  # Logs test results
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        log_update_frequency=log_update_frequency,
        val_interval=val_interval,
        sliding_average=sliding_average  # Or just: lambda x, w: np.mean(x[-w:])
    )

    print("trained:", model_path)
    # Shap
    # support_images, support_labels, query_images, query_labels, support_paths, query_paths = next(iter(test_loader)) # Get one episode

    # explainer = PrototypicalSHAPExplainer(class_names=["FK", "BK"], model=model, image_size=(288, 288))

    # # For each few-shot episode:
    # explainer.set_support_set(support_images, support_labels)
    # explainer.load_single_image("/Users/buddy/Documents/Vscode/opthalfsl/src2/dataset/test/1/KK11.jpg")

    # filename_model = os.path.basename(load_model) # just for save image name
    # filename_model = os.path.splitext(filename_model)[0]
    # filename_model = os.path.splitext(filename_model)[0]
    # print(filename_model)

    # shap_vals = explainer.explain(max_evals=1000)
    # explainer.save_explanation(shap_vals, filename_model=filename_model)
