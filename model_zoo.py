import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import csv
from sklearn.metrics import precision_score, recall_score
import os
import torch.nn.functional as F

# --- Model ---
class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder_head = nn.Sequential(
            nn.Linear(2048, 1024),  # EfficientNet-B0 output is 1280
            nn.ReLU(),
            nn.Linear(1024,256)
        )

    def forward(self, support_images, support_labels, query_images):
        z_support = self.encoder_head(self.backbone(support_images))
        z_query = self.encoder_head(self.backbone(query_images))
        n_way = len(torch.unique(support_labels))
        z_proto = torch.cat([
            z_support[torch.nonzero(support_labels == label).squeeze(1)].mean(0, keepdim=True)
            for label in range(n_way)
        ])
        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        return scores

class Trainer:
    def __init__(self, model, device, lr=0.0001, patience=5, model_path="models/best_model.pth",
                 log_file="logs/training_log.csv", result_file="results/evaluation_results.csv"):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopper = EarlyStopping(patience=patience, verbose=True, path=model_path)
        self.model_path = model_path
        self.log_file = log_file
        self.result_file = result_file

    def fit(self, support_images, support_labels, query_images, query_labels):
        self.model.train()
        self.optimizer.zero_grad()
        scores = self.model(support_images.to(self.device), support_labels.to(self.device), query_images.to(self.device))
        loss = self.criterion(scores, query_labels.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, val_loader):
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for support_images, support_labels, query_images, query_labels, _, _ in val_loader:
                scores = self.model(support_images.to(self.device), support_labels.to(self.device), query_images.to(self.device))
                loss = self.criterion(scores, query_labels.to(self.device))
                losses.append(loss.item())

                predictions = torch.argmax(scores, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(query_labels.cpu().numpy())
                correct_preds += (predictions == query_labels.to(self.device)).sum().item()
                total_preds += query_labels.size(0)

        avg_loss = np.mean(losses)
        accuracy = 100 * correct_preds / total_preds
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}")
        return avg_loss, accuracy, precision, recall

    def train(self, train_loader, val_loader, log_update_frequency=100, val_interval=500, sliding_average=lambda x, w: np.mean(x[-w:])):
        all_loss = []
        epoch_counter = 0
        # Write header for training log
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Validation Accuracy", "Precision", "Recall"])

            with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
                for episode_index, (support_images, support_labels, query_images, query_labels, _, _) in tqdm_train:
                    loss_value = self.fit(support_images, support_labels, query_images, query_labels)
                    all_loss.append(loss_value)

                    if episode_index % log_update_frequency == 0:
                        tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))

                    if episode_index % val_interval == 0 and episode_index > 0:
                        val_loss, val_acc, precision, recall = self.validate(val_loader)
                        avg_train_loss = np.mean(all_loss[-val_interval:])
                        epoch_counter += 1

                        writer.writerow([epoch_counter, avg_train_loss, val_loss, val_acc, precision, recall])

                        print(f"\nValidation Loss at episode {episode_index}: {val_loss:.4f}")
                        self.early_stopper(val_loss, self.model)

                        if self.early_stopper.early_stop:
                            print("Early stopping triggered.")
                            break


        # self.model.load_state_dict(torch.load(self.model_path))

    def evaluate_on_one_task(self, support_images, support_labels, query_images, query_labels):
        self.model.eval()
        with torch.no_grad():
            scores = self.model(support_images.to(self.device), support_labels.to(self.device), query_images.to(self.device))
            predictions = torch.argmax(scores, dim=1)
            correct = (predictions == query_labels.to(self.device)).sum().item()
            total = query_labels.size(0)
        return correct, total

    def evaluate(self, data_loader):

        total_preds = 0
        correct_preds = 0
        self.model.eval()

        with open(self.result_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["support_image_paths", "query_image_path", "probability", "true_label", "predicted_label"])

            with torch.no_grad():
                for support_images, support_labels, query_images, query_labels, support_paths, query_paths in tqdm(data_loader):
                    support_images = support_images.to(self.device)
                    query_images = query_images.to(self.device)
                    support_labels = support_labels.to(self.device)
                    query_labels = query_labels.to(self.device)

                    scores = self.model(support_images, support_labels, query_images)
                    probabilities = F.softmax(scores, dim=1)

                    support_filenames = ";".join(os.path.basename(p) for p in support_paths)

                    for i in range(len(query_labels)):
                        query_filename = os.path.basename(query_paths[i])
                        prob = probabilities[i][1].item()  # probability of class 1
                        pred_label = 1 if prob > 0.5 else 0  # threshold-based prediction
                        true_label = query_labels[i].item()
                        correct_preds += int(pred_label == true_label)
                        total_preds += 1

                        writer.writerow([support_filenames, query_filename, prob, true_label, pred_label])

        accuracy = 100 * correct_preds / total_preds if total_preds > 0 else 0
        print(f"\nEvaluation over {len(data_loader)} tasks: Accuracy = {accuracy:.2f}%")




# Dummy EarlyStopping class placeholder
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"Validation loss decreased. Saving model to {self.path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
