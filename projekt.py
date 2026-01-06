import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


device = torch.device("cpu")

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_CLASSES = 43
DATA_DIR = "./traffic_signs/archive"
TRAIN_DIR = os.path.join(DATA_DIR, 'Train')

CLASS_NAMES = {
    0: 'Speed limit 20', 1: 'Speed limit 30', 2: 'Speed limit 50',
    3: 'Speed limit 60', 4: 'Speed limit 70', 5: 'Speed limit 80',
    6: 'End speed limit 80', 7: 'Speed limit 100', 8: 'Speed limit 120',
    9: 'No passing', 10: 'No passing (trucks)', 11: 'Priority',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'No trucks', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware ice/snow',
    31: 'Wild animals crossing', 32: 'End limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout', 41: 'End no passing', 42: 'End no passing (trucks)'
}

# Data preparation
def prepare_data():
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Folder {TRAIN_DIR} does not exist!")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
    
    total_count = len(full_dataset)
    train_size = int(0.8 * total_count)
    val_size = total_count - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader, val_dataset

# MODEL
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 6 * 6, 120)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training
def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("\nTraining")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Visualization of the training process
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Development of Loss (Overfitting?)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(val_accuracies, label='Val Accuracy', marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Development Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./images/validation.png', dpi=300, bbox_inches='tight')
    plt.show()

# Confusion matrix analysis
def analyze_confusion_matrix(model, val_loader):
    print("\nConfusion matrix analysis")
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Overall accuracy
    acc = 100 * sum([p == t for p, t in zip(y_pred, y_true)]) / len(y_true)
    print(f"Overall accuracy: {acc:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # TOP 10 most confused pairs
    confusion_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'True': i,
                    'Pred': j,
                    'Count': cm[i, j],
                    'True_Name': CLASS_NAMES.get(i, f'Class {i}'),
                    'Pred_Name': CLASS_NAMES.get(j, f'Class {j}')
                })
    
    confusion_df = pd.DataFrame(confusion_pairs).sort_values('Count', ascending=False)
    
    print("\nTOP 10 most confused pairs:")
    print(confusion_df.head(10).to_string(index=False))
    
    # Vizualization confusion matrix (top 20 classes)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix (Top 20 classes)')
    plt.ylabel('Real class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.savefig('./images/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return confusion_df

# Robustness test
def apply_corruption(images, corruption_fn):
    # Denormalize from [-1,1] to [0,1]
    images = images * 0.5 + 0.5
    # Apply corruption
    images = corruption_fn(images)
    # Clip to [0,1]
    images = torch.clamp(images, 0, 1)
    # Renormalize to [-1,1]
    images = (images - 0.5) / 0.5
    return images

def robustness_test(model, val_loader):
    print("\nRobustness test (Real-World Conditions)")
    
    experiments = {
        'Baseline': lambda x: x,
        'Light Blur': lambda x: transforms.functional.gaussian_blur(x, kernel_size=3),
        'Heavy Blur': lambda x: transforms.functional.gaussian_blur(x, kernel_size=5),
        'Gaussian Noise (σ=0.1)': lambda x: x + torch.randn_like(x) * 0.1,
        'Gaussian Noise (σ=0.2)': lambda x: x + torch.randn_like(x) * 0.2,
        'Brightness -30%': lambda x: x * 0.7,
        'Brightness -50%': lambda x: x * 0.5,
        'Low Contrast': lambda x: 0.5 + (x - 0.5) * 0.5,
    }
    
    results = {}
    model.eval()
    
    for name, corruption_fn in experiments.items():
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                if name != 'Baseline':
                    images = apply_corruption(images, corruption_fn)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        results[name] = acc
        drop = acc - results['Baseline'] if name != 'Baseline' else 0
        print(f"{name:25} -> Acc: {acc:5.2f}% (Drop: {drop:+5.2f}%)")
    
    # Vizualization
    plt.figure(figsize=(14, 7))
    names = list(results.keys())
    values = list(results.values())
    colors = ['green' if v > 90 else 'orange' if v > 70 else 'red' for v in values]
    bars = plt.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
    
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Robustness: How does the model handle real-world conditions?', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.axhline(y=90, color='g', linestyle='--', label='Good (>90%)')
    plt.axhline(y=70, color='orange', linestyle='--', label='Acceptable (>70%)')
    plt.legend()
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./images/robustness.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# Grad-CAM vizualization
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, class_idx):
        self.model.eval()
        output = self.model(input_image)
        
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam

def visualize_gradcam(model, val_dataset, num_samples=6):
    print("\nGrad-CAM vizualization")
    grad_cam = GradCAM(model, model.conv2)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(18, 6))
    
    for i in range(num_samples):
        # Random sample
        idx = np.random.randint(0, len(val_dataset))
        image, true_label = val_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            output = model(image_batch)
            _, predicted = torch.max(output, 1)
            pred_label = predicted.item()
        
        # Grad-CAM
        image_batch.requires_grad = True
        cam = grad_cam.generate_cam(image_batch, pred_label)
        
        # Denormalize image for visualization
        img_show = image.cpu() * 0.5 + 0.5
        img_show = img_show.permute(1, 2, 0).numpy()
        
        # Resize CAM
        cam_np = cam.squeeze().cpu().numpy()
        
        cam_tensor = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(32, 32), mode='bilinear', align_corners=False)
        cam_resized = cam_resized.squeeze().numpy()
        
        # Overlay
        axes[0, i].imshow(img_show)
        axes[0, i].set_title(f"Original\nTrue: {true_label}\nPred: {pred_label}", fontsize=9)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(img_show)
        axes[1, i].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[1, i].set_title("Grad-CAM\n(red=focus)", fontsize=9)
        axes[1, i].axis('off')
    
    plt.suptitle('Grad-CAM: What is the model looking at?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./images/Grad-CAM.png', dpi=300, bbox_inches='tight')
    plt.show()
    
       
# Adversarial robustness (FGSM)
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

def adversarial_test(model, val_loader):
    print("\nAdversarial robustness (FGSM Attack)")
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
    accuracies = []
    criterion = nn.CrossEntropyLoss()
    
    for eps in epsilons:
        correct = 0
        total = 0
        
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            
            if eps > 0:
                perturbed = fgsm_attack(images, eps, images.grad.data)
            else:
                perturbed = images
            
            with torch.no_grad():
                outputs = model(perturbed)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        accuracies.append(acc)
        print(f"ε = {eps:.2f} -> Accuracy: {acc:.2f}%")
    
    # Vizualization
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epsilon (size of perturbation)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Adversarial Robustness (FGSM Attack)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Safe threshold')
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random guess')
    plt.legend()
    
    for i, (e, a) in enumerate(zip(epsilons, accuracies)):
        plt.text(e, a + 2, f'{a:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('./images/adversial_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("TRAFFIC SIGN CLASSIFICATION")
    
    # 1. Data
    train_loader, val_loader, val_dataset = prepare_data()
    
    # 2. Model
    model = TrafficSignNet(num_classes=NUM_CLASSES).to(device)
    
    # 3. Training
    train_model(model, train_loader, val_loader, epochs=EPOCHS)
    
    # 4. Confusion Matrix analysis
    confusion_df = analyze_confusion_matrix(model, val_loader)
    
    # 5. Robustness test
    robustness_results = robustness_test(model, val_loader)
    
    # 6. Grad-CAM vizualization
    visualize_gradcam(model, val_dataset, num_samples=6)
    
    # 7. Adversarial robustness
    adversarial_test(model, val_loader)
    
