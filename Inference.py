'''
python inference.py --dataset dataset.pickle --model unet_50_epochs.pth

'''

import argparse
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, TensorDataset
import warnings

# ignore FutureWarning torch
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#%% preprocessing
# same steps as for training
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


def calculate_dice(pred_mask, true_mask):
    pred_mask = pred_mask > 0.5 
    intersection = torch.sum(pred_mask * true_mask)
    dice = (2.0 * intersection) / (torch.sum(pred_mask) + torch.sum(true_mask) + 1e-7)
    return dice.item()


def calculate_iou(pred_mask, true_mask):
    pred_mask = pred_mask > 0.5
    intersection = torch.logical_and(pred_mask, true_mask)
    union = torch.logical_or(pred_mask, true_mask)
    if torch.sum(union) == 0:
        return 1.0
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    return iou.item()

def calculate_metrics(outputs, masks):
    ious = []
    dices = []
    with torch.no_grad():
        for i in range(outputs.size(0)):
            pred_mask = outputs[i, 0].cpu()
            true_mask = masks[i].cpu()
            
            iou = calculate_iou(pred_mask, true_mask)
            dice = calculate_dice(pred_mask, true_mask)
            
            ious.append(iou)
            dices.append(dice)
    return np.mean(ious), np.mean(dices)

# Main function to run inference
def main(dataset_path, model_path):
    logger.info("Loading dataset...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    val_data = data['val']
    logger.info(f"Validation data keys: {val_data[0].keys()}")
    logger.info(f"Image shape: {val_data[0]['image'].shape}, Mask shape: {val_data[0]['mask'].shape}")

    X_val = [pad_image(preprocess_image(entry['image'].transpose(2, 0, 1))) for entry in val_data]  # Fix transposition
    y_val = [torch.tensor(entry['mask']) for entry in val_data]

    val_dataset = TensorDataset(torch.stack(X_val), torch.stack(y_val))
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Load the model and set it to evaluation mode (on CPU)
    logger.info("Loading model...")
    model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1, activation='sigmoid')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Run inference and calculate IoU and Dice
    logger.info("Running inference on CPU...")
    ious = []
    dices = []
    for images, masks in val_loader:
        images = images.float()
        outputs = model(images)
        iou, dice = calculate_metrics(outputs, masks)
        ious.append(iou)
        dices.append(dice)

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)
    logger.info(f"Mean IoU on validation set: {mean_iou:.4f}")
    logger.info(f"Mean Dice on validation set: {mean_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on validation set using a pretrained U-Net model")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset pickle file")
    parser.add_argument('--model', type=str, required=True, help="Path to the saved model file")
    
    args = parser.parse_args()
    main(args.dataset, args.model)

#python inference.py --dataset dataset.pickle --model unet_building_segmentation_50.pth 