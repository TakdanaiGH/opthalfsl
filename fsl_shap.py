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

    # --- Datasets ---
    base_path = "/Users/buddy/Documents/Vscode/opthalfsl/src2/dataset"

    # --- Parameters ---
    test_image_size = 384
    N_WAY = 2
    N_SHOT = 3
    N_QUERY = 3
    N_EVALUATION_TASKS = 100 # Test
    log_update_frequency = 50
    val_interval = 20

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize([int(test_image_size), int(test_image_size)]),
        transforms.CenterCrop(test_image_size),
        transforms.ToTensor(),
    ])

    # --- Datasets ---
    base_path = "/Users/buddy/Documents/Vscode/opthalfsl/src2/dataset"

    test_set = ImageFolderWithPaths(os.path.join(base_path, "test"), transform=test_transform)
    test_set.labels = test_set.targets
    test_set.get_labels = lambda: test_set.targets    # <- ADD THIS

    test_sampler = TaskSampler(test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS)
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
        # persistent_workers=True,
        # prefetch_factor=4,
    )

    backbone_name = "resnet50.a1_in1k"
    # Load backbone
    # cnn_backbone = resnet18(pretrained=True) # 512
    # cnn_backbone = resnet50(pretrained=True) # 2048
    # cnn_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) # 1000
    # cnn_backbone = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in12k', pretrained=True, num_classes=0) # 768
    # cnn_backbone = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0) # 2048
    cnn_backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0) # 1280
    # cnn_backbone = timm.create_model('timm/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k', pretrained=True, num_classes=0) # 512 # 384
    # cnn_backbone = timm.create_model('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384', pretrained=True, num_classes=0) # 1536 # 256 images size

    cnn_backbone.fc = nn.Identity()

    for param in cnn_backbone.parameters():
        param.requires_grad = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = PrototypicalNetworks(cnn_backbone).to(device)
    load_model = "models/best_resnet50_100e_448.a1_in1k.pth"

    print("loading model:", load_model)
    model.load_state_dict(torch.load(load_model))

    # Shap
    support_images, support_labels, query_images, query_labels, support_paths, query_paths = next(iter(test_loader)) # Get one episode

    explainer = PrototypicalSHAPExplainer(class_names=["FK", "BK"], model=model, image_size=(288, 288))

    # For each few-shot episode:
    explainer.set_support_set(support_images, support_labels)
    explainer.load_single_image("/Users/buddy/Documents/Vscode/opthalfsl/src2/dataset/test/1/RAMA1.jpg")

    filename_model = os.path.basename(load_model) # just for save image name
    filename_model = os.path.splitext(filename_model)[0]
    filename_model = os.path.splitext(filename_model)[0]
    print(filename_model)

    shap_vals = explainer.explain(max_evals=1000)
    explainer.save_explanation(shap_vals, filename_model=filename_model)
