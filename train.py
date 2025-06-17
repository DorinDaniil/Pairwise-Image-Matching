import torch
import numpy as np
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms

def calculate_accuracy(probabilities, labels):
    predicted = (probabilities > 0.5).float()
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

def calculate_metrics(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
    return f1, recall, precision

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_optimizer(net, config):
    optimizer_name = config['optimizer']['name']
    if optimizer_name == 'SGD':
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['optimizer']['weight_decay']
        )
    elif optimizer_name == 'Adam':
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    elif optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=config['optimizer']['lr'],
            betas=config['optimizer']['betas'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if config['optimizer'].get('load_state_dict_path'):
        opt.load_state_dict(torch.load(config['optimizer']['load_state_dict_path']))
    return opt

def get_scheduler(opt, config):
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=config['scheduler']['milestones'],
        gamma=config['scheduler']['gamma']
    )
    if config['scheduler'].get('load_state_dict_path'):
        sched.load_state_dict(torch.load(config['scheduler']['load_state_dict_path']))
    return sched

def save_checkpoint(model, optimizer, scheduler, epoch, config):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    checkpoint_path = os.path.join(config['model']['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Checkpoint loaded from epoch {epoch}')
        return epoch
    else:
        print('No checkpoint found. Starting training from scratch.')
        return 0

def train_model(model, train_loader, val_loader, config, resume=False):
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    if config.get('train_encoder', False):
        regularizer = torch.nn.CosineEmbeddingLoss(margin=config['regularizer']['margin'])

    num_epochs = config['training']['num_epochs']
    device = torch.device(config['training']['device'])
    writer = SummaryWriter(log_dir=config['data']['tensorboard_logdir'])
    checkpoint_interval = config['training']['checkpoint_interval']
    checkpoint_dir = config['model']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)

    start_epoch = 0
    if resume:
        latest_checkpoint = max(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)],
            key=os.path.getctime
        )
        start_epoch = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_total = 0
        all_train_labels = []
        all_train_predictions = []

        for batch1, batch2 in tqdm(train_loader):
            batch1, batch2 = batch1.to(device), batch2.to(device)
            batch_size = batch1.size(0)
            pairs_batch_size = batch_size ** 2

            optimizer.zero_grad()

            logits = model(batch1, batch2)

            labels = torch.eye(batch_size).to(device).view(-1)
            weight = torch.full_like(labels, 0.3, dtype=torch.float)
            weight[labels == 1] = 0.7

            loss = torch.nn.functional.binary_cross_entropy(logits.view(-1), labels, weight=weight)

            if config.get('train_encoder', False):
                v1, v2 = model.get_q(batch1), model.get_q(batch2)
                additional_loss = regularizer(v1, v2, torch.ones(batch_size, device=device))
                total_loss = loss + config['regularizer']['lambda'] * additional_loss
            else:
                total_loss = loss

            total_loss.backward()
            
            # total_norm = 0.0
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         param_norm = param.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f'Gradient norm at epoch {epoch+1}, batch {batch1.size(0)}: {total_norm:.4f}')

            optimizer.step()
            scheduler.step()
            train_loss += total_loss.item() * pairs_batch_size
            train_total += pairs_batch_size

            model.eval()
            with torch.no_grad():
                probabilities = model(batch1, batch2).view(-1)
                all_train_labels.extend(labels.cpu().numpy())
                all_train_predictions.extend(probabilities.cpu().numpy())
            model.train()

        train_loss /= train_total
        train_accuracy = calculate_accuracy(torch.tensor(all_train_predictions), torch.tensor(all_train_labels))
        train_f1, train_recall, train_precision = calculate_metrics(all_train_labels, np.round(all_train_predictions))

        model.eval()
        val_loss = 0.0
        val_total = 0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad():
            for batch1, batch2 in val_loader:
                batch1, batch2 = batch1.to(device), batch2.to(device)
                batch_size = batch1.size(0)
                pairs_batch_size = batch_size ** 2

                logits = model(batch1, batch2)

                labels = torch.eye(batch_size).to(device).view(-1)
                weight = torch.full_like(labels, 0.3, dtype=torch.float)
                weight[labels == 1] = 0.7

                loss = torch.nn.functional.binary_cross_entropy(logits.view(-1), labels, weight=weight)

                if config.get('train_encoder', False):
                    v1, v2 = model.get_q(batch1), model.get_q(batch2)
                    additional_loss = regularizer(v1, v2, torch.ones(batch_size, device=device))
                    total_loss = loss + config['regularizer']['lambda'] * additional_loss
                else:
                    total_loss = loss

                val_loss += total_loss.item() * pairs_batch_size
                val_total += pairs_batch_size

                probabilities = model(batch1, batch2).view(-1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(probabilities.cpu().numpy())

        val_loss /= val_total
        val_accuracy = calculate_accuracy(torch.tensor(all_val_predictions), torch.tensor(all_val_labels))
        val_f1, val_recall, val_precision = calculate_metrics(all_val_labels, np.round(all_val_predictions))

        # Вывод текущего значения learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, \
              Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Train Recall: {train_recall:.4f}, \
              Train Precision: {train_precision:.4f}, Val Loss: {val_loss:.4f}, \
              lr: {current_lr}, \
              Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}, \
              Val Precision: {val_precision:.4f}')

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('Recall/Train', train_recall, epoch)
        writer.add_scalar('Precision/Train', train_precision, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_accuracy, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)
        writer.add_scalar('Recall/Val', val_recall, epoch)
        writer.add_scalar('Precision/Val', val_precision, epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, config)

    writer.close()
