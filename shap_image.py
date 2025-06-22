import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class PrototypicalSHAPExplainer:
    def __init__(self, model, class_names=None, normalize=True, image_size=(224, 224)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        self.class_names = class_names
        self.normalize = normalize
        self.image_size = image_size

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.support_images = None
        self.support_labels = None
        self.X = None
        self.filename = None
        self.explainer = None

    def set_support_set(self, support_images: torch.Tensor, support_labels: torch.Tensor):
        self.support_images = support_images.to(self.device)
        self.support_labels = support_labels.to(self.device)

        if self.class_names is None:
            unique_labels = sorted(torch.unique(support_labels).cpu().tolist())
            self.class_names = [f"class_{i}" for i in unique_labels]

    def load_single_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size)
        image_np = np.array(image)  # shape: (H, W, 3)
        self.X = np.expand_dims(image_np, axis=0)  # shape: (1, H, W, 3)
        self.filename = image_path

        self.masker = shap.maskers.Image(mask_value=0.0, shape=self.X[0].shape)
        self.explainer = shap.Explainer(self._predict_fn, self.masker, output_names=self.class_names)

    def _predict_fn(self, X_batch):
        X_tensor = torch.tensor(X_batch, device=self.device, dtype=torch.float32)
        X_tensor = X_tensor.permute(0, 3, 1, 2) / 255.0  # Convert to NCHW and scale to [0, 1]

        if self.normalize:
            X_tensor = (X_tensor - self.mean) / self.std

        with torch.no_grad():
            scores = self.model(
                query_images=X_tensor,
                support_images=self.support_images,
                support_labels=self.support_labels
            )
        return scores.cpu().numpy()

    def get_prediction(self):
        if self.X is None:
            raise ValueError("No image loaded.")

        logits = self._predict_fn(self.X).squeeze()
        exp_scores = np.exp(logits - np.max(logits))
        probs = exp_scores / exp_scores.sum()

        pred_class_idx = int(np.argmax(probs))
        pred_class_name = self.class_names[pred_class_idx]

        return pred_class_name, probs

    def explain(self, max_evals):
        if self.X is None:
            raise ValueError("No image loaded. Use load_single_image() first.")
        return self.explainer(self.X, max_evals=max_evals)

    def save_explanation(self, shap_values, filename_model="default", output_dir="shap_outputs"):
        if self.X is None or self.filename is None:
            raise ValueError("No image loaded. Call load_single_image() first.")

        pred_class, probs = self.get_prediction()
        prob_str = ", ".join(f"{name}: {p:.2f}" for name, p in zip(self.class_names, probs))

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(self.filename)
        base_name = os.path.splitext(base_name)[0]
        save_path = os.path.join(output_dir, f"shap_{base_name}_{filename_model}.png")

        plt.figure(figsize=(6, 6))
        shap.image_plot(shap_values, self.X, show=False)
        plt.suptitle(f"File: {base_name}\nPrediction: {pred_class}\nProbs: {prob_str}", fontsize=12)
        plt.savefig(save_path)
        plt.close()

        print(f"[SHAP] Saved explanation to {save_path}")
