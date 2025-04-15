import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config,
    AutoFeatureExtractor
)
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch.backends.cudnn as cudnn
from transformers import Wav2Vec2FeatureExtractor
from VAD_dataset import VADDataset
import math
# subsets for testing
import random
from torch.utils.data import Subset
# Import your new modules
from model import *
import wandb

# Enable CuDNN Benchmarking
cudnn.benchmark = True

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return local_rank, world_size

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()



def setup_model(local_rank, layer=14):    
    model = Wav2VecWithClassifier()

    model.to(local_rank)
    return model

def calculate_class_weights(dataset, batch_size=64):
    print("Calculating class weights...")
    speech_count = 0
    total_count = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing dataset"):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_labels = []
        
        for idx in batch_indices:
            labels = dataset[idx]["labels"]
            batch_labels.extend(labels.numpy())
            
        batch_labels = np.array(batch_labels)
        speech_count += np.sum(batch_labels > 0)
        total_count += len(batch_labels)
        
        del batch_labels
    
    speech_ratio = speech_count / total_count
    nonspeech_ratio = 1 - speech_ratio
    
    weights = torch.tensor([
        1/(nonspeech_ratio + 1e-5),
        1/(speech_ratio + 1e-5)
    ], dtype=torch.float32)
    
    weights = weights / weights.sum() * 2
    return weights

def reduce_tensor(tensor, world_size):
    if dist.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
        return rt
    else:
        return tensor  

def calculate_metrics(predictions, labels, world_size):
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    true_positive = torch.tensor(np.sum((predictions == 1) & (labels == 1)), dtype=torch.float32).cuda()
    false_positive = torch.tensor(np.sum((predictions == 1) & (labels == 0)), dtype=torch.float32).cuda()
    true_negative = torch.tensor(np.sum((predictions == 0) & (labels == 0)), dtype=torch.float32).cuda()
    false_negative = torch.tensor(np.sum((predictions == 0) & (labels == 1)), dtype=torch.float32).cuda()
    
    if dist.is_initialized():
        true_positive = reduce_tensor(true_positive, world_size)
        false_positive = reduce_tensor(false_positive, world_size)
        true_negative = reduce_tensor(true_negative, world_size)
        false_negative = reduce_tensor(false_negative, world_size)
    
    true_positive = true_positive.cpu().item()
    false_positive = false_positive.cpu().item()
    true_negative = true_negative.cpu().item()
    false_negative = false_negative.cpu().item()
    
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'speech_ratio': np.mean(labels == 1)
    }


def train_one_epoch(
    model, dataloader, optimizer, local_rank,
    class_weights, world_size,
    scheduler=None
    ):

    model.train()
    
    total_loss = torch.tensor(0.0).to(local_rank)
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training") if local_rank == 0 else dataloader
    for batch in progress_bar:
        input_values = batch["input_values"].to(local_rank) # -> (64, 99)
        labels = batch["labels"].to(local_rank) # -> (64, 32000)
        # print(f"Labels: {labels.shape}")
        optimizer.zero_grad()
        # print("Inputing vals in model")
        outputs = model(input_values=input_values)
        logits = outputs["logits"].contiguous()  # -> (64, 99, 2)
        # print(f"logits: {logits.shape}")

        # Prepare labels for frame-level classification
        n_frames = logits.shape[1]
        frame_size = input_values.shape[1] // n_frames
        # print(f"Frame size: {frame_size}")
        labels = labels.float().contiguous()
        labels = labels.unfold(1, frame_size, frame_size)[:, :n_frames].mean(dim=-1) > 0.5
        labels = labels.long().contiguous()  # -> (64, 99)
        # print(f"labels: {labels.shape}")

        predictions = torch.argmax(logits, dim=-1) # -> (64, 99)

        # print(f"CE: logits {logits.reshape(-1, 2).shape} labels {labels.reshape(-1).shape}")

        # CE: logits torch.Size([6336, 2]) labels torch.Size([6336])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1),
            weight=class_weights
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        all_preds.extend(predictions.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        total_loss += loss.item()

        if local_rank == 0 and isinstance(progress_bar, tqdm):
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    
    if dist.is_initialized():
        total_loss = reduce_tensor(total_loss, world_size)

    avg_loss = total_loss.item() / len(dataloader)
    metrics = calculate_metrics(all_preds, all_labels, world_size)
    
    return avg_loss, metrics



def collate_fn(batch):
    input_values = torch.stack([x["input_values"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {
        "input_values": input_values,
        "labels": labels,
        "audio_path": [x["audio_path"] for x in batch],
        "num_speakers": [x["num_speakers"] for x in batch],
        "window_idx": [x["window_idx"] for x in batch],
        "start_time": [x["start_time"] for x in batch],
        "end_time": [x["end_time"] for x in batch],
    }


def validate(model, dataloader, local_rank, class_weights, world_size):
    model.eval()

    total_loss = torch.tensor(0.0).to(local_rank)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation") if local_rank == 0 else dataloader
        for batch_idx, batch in enumerate(progress_bar):
            input_values = batch["input_values"].to(local_rank)
            labels = batch["labels"].to(local_rank)
            
            outputs = model(input_values=input_values)
            logits = outputs["logits"].contiguous()  # Shape: [frame, B, num_labels]
            
            labels = labels.float().contiguous()
            n_frames = logits.shape[1]
            frame_size = input_values.shape[1] // n_frames
            labels = labels.unfold(1, frame_size, frame_size)[:, :n_frames].mean(dim=-1) > 0.5
            labels = labels.long().contiguous()
            
            predictions = torch.argmax(logits, dim=-1)

            # print("Input:", labels)
            # print("PRED:", predictions)           
            
            # print("Input:", labels.reshape(-1).shape)
            # print("PRED:", logits.reshape(-1, 2).shape)
            
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2),
                labels.reshape(-1),
                weight=class_weights
            )
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            if local_rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Reduce loss across all GPUs
    if dist.is_initialized():
        total_loss = reduce_tensor(total_loss, world_size)
    
    avg_loss = total_loss.item() / len(dataloader)
    metrics = calculate_metrics(all_preds, all_labels, world_size)
        
    return avg_loss, metrics

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Wav2Vec Feature Extraction")
    parser.add_argument(
        "--layer", 
        type=int, 
        default=14,  # Default layer value
        help="Specify which layer of the Wav2Vec model to use for feature extraction"
    )

    parser.add_argument(
        "--test", 
        type=bool, 
        default=False,  # Default layer value
        help="Specificy when you are testing the code"
    )
    return parser.parse_args()

def main():
    args = get_args()
    layer = int(args.layer)
    is_test = bool(args.test)

    print(f"Using layer: {layer} and test: {is_test}")

    local_rank, world_size = setup_distributed()
    
    if local_rank == 0:
        print(f"Training with {world_size} GPUs")
        # w2v-training-dummy-dataset
        if is_test:
            project_name = "test"
        else:
            project_name = 'w2v_train_umd_and_cb_both_robust_large'
        wandb.init(
            project=project_name,
            config={
                "layer": layer,
                "max_duration": 2.0,
                "stride_duration": 0.25,
                "batch_size": 64,
                "learning_rate": 1e-4,
                "weight_decay": 1e-2,
                "num_epochs": 100,
                'class_weights': (1.5, 0.5)
            }
        )
    
    # ------------------ load model ------------------ #
    model = setup_model(local_rank, layer=layer)
    # ------------------------------------------------------#


    # Wrap in DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust")

    # dataset creation
    max_duration = 2.0
    stride_duration = 0.25
    
    full_train_dataset = VADDataset(
        manifest_path="./train.json",
        feature_extractor=feature_extractor,
        max_duration_s=max_duration,
        stride_duration_s=stride_duration
    )
    
    full_val_noisy_dataset = VADDataset(
        manifest_path="./dev_noisy.json",
        feature_extractor=feature_extractor,
        max_duration_s=max_duration,
        stride_duration_s=stride_duration
    )

    full_val_denoised_dataset = VADDataset(
        manifest_path="./dev_denoised.json",
        feature_extractor=feature_extractor,
        max_duration_s=max_duration,
        stride_duration_s=stride_duration
    )


    if is_test:
        full_train_dataset = Subset(full_train_dataset, range(10000))
        full_val_noisy_dataset = Subset(full_val_noisy_dataset, range(500))
        full_val_denoised_dataset = Subset(full_val_denoised_dataset, range(500))


    class_weights = torch.tensor([1.5, 0.5]).to(local_rank)

    train_sampler = DistributedSampler(full_train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        full_train_dataset,
        batch_size=64,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_noisy_sampler = DistributedSampler(full_val_noisy_dataset, shuffle=False) if world_size > 1 else None
    val_noisy_loader = DataLoader(
        full_val_noisy_dataset,
        batch_size=64,
        shuffle=False,
        sampler=val_noisy_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_denoised_sampler = DistributedSampler(full_val_denoised_dataset, shuffle=False) if world_size > 1 else None
    val_denoised_loader = DataLoader(
        full_val_denoised_dataset,
        batch_size=64,
        shuffle=False,
        sampler=val_denoised_sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    # Training loop
    num_epochs = 35

    best_noisy_f1 = 0
    best_denoised_f1 = 0
    epochs_without_improvement = 0
    patience = 5
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    stop_flag = torch.tensor([0], device=local_rank)

    try:
        for epoch in range(num_epochs):
            if local_rank == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # train
            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                local_rank,
                class_weights,
                world_size,
                scheduler=scheduler   
            )

            current_lr = optimizer.param_groups[0]['lr']
            
            if local_rank == 0:
                wandb.log({
                    "train/loss": train_loss,
                    "train/f1": train_metrics['f1'],
                    "learning_rate": current_lr,
                    "epoch": epoch+1

                })

            scheduler.step(train_loss)

            if local_rank == 0:
                print(f"Training - Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}")
            
            # validation
            if epoch % 3 == 0:
                val_noisy_loss, val_noisy_metrics = validate(model, val_noisy_loader, local_rank, class_weights, world_size)
                val_denoised_loss, val_denoised_metrics = validate(model, val_denoised_loader, local_rank, class_weights, world_size)

                if local_rank == 0:
                    print(f"Validation Noisy - Loss: {val_noisy_loss:.4f}, F1: {val_noisy_metrics['f1']:.4f}")
                    print(f"Validation Denoised - Loss: {val_denoised_loss:.4f}, F1: {val_denoised_metrics['f1']:.4f}")
                    
                    wandb.log({
                        "val_noisy/loss": val_noisy_loss,
                        "val_noisy/f1": val_noisy_metrics['f1'],
                        "val_denoised/loss": val_denoised_loss,
                        "val_denoised/f1": val_denoised_metrics['f1'],
                    })

                    noisy_improvement = False
                    denoised_improvement = False
                    if val_noisy_metrics['f1'] > best_noisy_f1:
                        noisy_improvement = True
                        best_noisy_f1 = val_noisy_metrics['f1']
                        epochs_without_improvement = 0
                        if not is_test:
                            torch.save(model.state_dict(), f"./checkpoints-noisy/ckpt-{epoch+1}.pt")
                        wandb.log({
                            "best-noisy/best_val_noisy_f1": best_noisy_f1,
                            'best-noisy/best_val_noisy_f1_loss': val_noisy_loss
                        })
                    
                    if val_denoised_metrics['f1'] > best_denoised_f1:
                        denoised_improvement = True

                        best_denoised_f1 = val_denoised_metrics['f1']
                        epochs_without_improvement = 0
                        if not is_test:
                            torch.save(model.state_dict(), f"./checkpoints-denoised/ckpt-{epoch+1}.pt")
                        wandb.log({
                            "best-denoised/best_val_denoised_f1": best_denoised_f1,
                            'best-denoised/best_val_denoised_f1_loss': val_denoised_loss
                        })

                    if noisy_improvement == False and denoised_improvement == False:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            print("Early stopping triggered.")
                            wandb.log({"early_stopping": epoch + 1})
                            stop_flag[0] = 1

            if dist.is_initialized():
                dist.broadcast(stop_flag, src=0)

            # all ranks see stop_flag now:
            if stop_flag.item() == 1:
                break

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if local_rank == 0:
            wandb.finish()
        raise e
    finally:
        if local_rank == 0:
            wandb.finish()
        cleanup()
        import sys
        sys.exit(0)

if __name__ == "__main__":
    main()
