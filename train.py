''' USAGE
python train.py --batch_size 8 --epochs 1--learning_rate 0.0001 --model_save_path unet_model.pth --dataset_path dataset.pickle 
'''

import argparse
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import segmentation_models_pytorch as smp
import torch.optim as optim
import cv2
from sklearn.model_selection import train_test_split
import warnings

# Setting up argparse to handle input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a U-Net model on custom dataset.")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--model_save_path', type=str, default="unet_model.pth", help='Path to save the trained model.')
    parser.add_argument('--dataset_path', type=str, default='dataset.pickle', help='Path to the dataset pickle file.')
    return parser.parse_args()

def main():
    # Parse the arguments
    args = parse_args()

    # Logger setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Loading dataset...")

    # Load the dataset from pickle file
    with open(args.dataset_path, 'rb') as f:
        data = pickle.load(f)

    train_data = data['train']
    val_data = data['val']

    logger.info(f"Training data keys: {train_data[0].keys()}")
    logger.info(f"Image shape: {train_data[0]['image'].shape}, Mask shape: {train_data[0]['mask'].shape}")

    # Preprocessing functions
    def preprocess_image(image):
        max_value = np.max(image)
        if max_value == 0:
            return image
        return image / max_value  

    def pad_image(image):
        _, h, w = image.shape
        pad_h = (32 - h % 32) if h % 32 != 0 else 0
        pad_w = (32 - w % 32) if w % 32 != 0 else 0
        padded_image = F.pad(torch.tensor(image, dtype=torch.float32), (0, pad_w, 0, pad_h), mode='constant', value=0)
        return padded_image

    logger.info("Preprocessing and padding images...")
    X_train = [pad_image(preprocess_image(entry['image'].transpose(2, 0, 1))) for entry in train_data]  
    y_train = [torch.tensor(entry['mask']) for entry in train_data] 
    X_val = [pad_image(preprocess_image(entry['image'].transpose(2, 0, 1))) for entry in val_data]
    y_val = [torch.tensor(entry['mask']) for entry in val_data]

    logger.info(f"Number of training samples: {len(X_train)}, Number of validation samples: {len(X_val)}")

    # Convert the data into PyTorch Datasets
    logger.info("Creating data loaders...")
    train_dataset = TensorDataset(torch.stack(X_train), torch.stack(y_train))
    val_dataset = TensorDataset(torch.stack(X_val), torch.stack(y_val))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model Definition (U-Net using pre-trained ResNet-34 encoder)
    logger.info("Initializing U-Net model with ResNet34 encoder...")
    model = smp.Unet(encoder_name='resnet34',  
                     encoder_weights='imagenet', 
                     in_channels=3, 
                     classes=1,  
                     activation='sigmoid')  

    criterion = torch.nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  

    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Training on device: {device}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}...")
        
        for images, masks in train_loader:
            images = images.to(device).float()  
            masks = masks.to(device).float()  

            optimizer.zero_grad()  
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    # Eval
    logger.info("Starting model evaluation...")

    def calculate_iou(pred_mask, true_mask):
        pred_mask = pred_mask > 0.5
        intersection = torch.logical_and(pred_mask, true_mask)
        union = torch.logical_or(pred_mask, true_mask)
        if torch.sum(union) == 0:
            return 1.0
        return torch.sum(intersection).float() / torch.sum(union).float()

    def calculate_dice(pred_mask, true_mask):
        pred_mask = pred_mask > 0.5
        intersection = torch.sum(pred_mask * true_mask)
        return (2.0 * intersection) / (torch.sum(pred_mask) + torch.sum(true_mask) + 1e-7)

    model.eval()
    ious, dices = [], []

    with torch.no_grad():  
        for images, masks in val_loader:
            images = images.to(device).float()  
            masks = masks.to(device).float()  
            outputs = model(images)
            for i in range(outputs.size(0)):
                pred_mask = outputs[i, 0].cpu()
                true_mask = masks[i].cpu()

                iou = calculate_iou(pred_mask, true_mask)
                ious.append(iou.item())

                dice = calculate_dice(pred_mask, true_mask)
                dices.append(dice.item())

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)

    logger.info(f'Mean IoU: {mean_iou:.4f}')
    logger.info(f'Mean Dice Coefficient: {mean_dice:.4f}')

    # Save the model
    model_save_path = f"unet_{args.epochs}_epochs.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
