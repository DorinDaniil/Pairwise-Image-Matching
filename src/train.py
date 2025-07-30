import torch
import numpy as np
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ------- Метрики: инициализация, обновление, вычисление -------

def init_metrics():
    return {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

def update_metrics(metrics_dict, true_labels, predicted_labels):
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(predicted_labels, torch.Tensor):
        predicted_labels = predicted_labels.cpu().numpy()

    tp = ((predicted_labels == 1) & (true_labels == 1)).sum()
    fp = ((predicted_labels == 1) & (true_labels == 0)).sum()
    tn = ((predicted_labels == 0) & (true_labels == 0)).sum()
    fn = ((predicted_labels == 0) & (true_labels == 1)).sum()

    metrics_dict['tp'] += int(tp)
    metrics_dict['fp'] += int(fp)
    metrics_dict['tn'] += int(tn)
    metrics_dict['fn'] += int(fn)

    return metrics_dict

def compute_metrics(metrics_dict):
    tp = metrics_dict['tp']
    fp = metrics_dict['fp']
    tn = metrics_dict['tn']
    fn = metrics_dict['fn']

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall > 0) else 0.0

    return {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }

# ------- Загрузка конфигурации и оптимизаторы -------

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_optimizer(net, config):
    optimizer_name = config['optimizer']['name']
    if optimizer_name == 'Adam':
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

    return opt

def get_scheduler(opt, config):
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=config['scheduler']['milestones'],
        gamma=config['scheduler']['gamma']
    )
    return sched

def save_checkpoint(model, optimizer, scheduler, epoch, config):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
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

# ------- Основной тренировочный цикл -------

def train_model(model, train_loader, val_loader, config, resume=False):
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    if config['training']['contrastive_regularizer']:
        regularizer = torch.nn.CosineEmbeddingLoss(margin=config['regularizer']['margin'])

    num_epochs = config['training']['num_epochs']
    device = torch.device(config['training']['device'])
    checkpoint_interval = config['training']['checkpoint_interval']
    checkpoint_dir = config['training']['checkpoint_dir']
    writer = SummaryWriter(log_dir=config['data']['tensorboard_logdir'])
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
        train_metrics_sum = init_metrics()

        for batch1, batch2 in tqdm(train_loader):
            batch1, batch2 = batch1.to(device), batch2.to(device)
            batch_size = batch1.size(0)
            pairs_batch_size = batch_size ** 2

            optimizer.zero_grad()

            probabilities = model(batch1, batch2).view(-1)
            labels = torch.eye(batch_size).to(device).view(-1) # диагональ → 1, остальное 0

            # ошибка на 0 классе дороже, нужен precision = 1
            weight = torch.full_like(labels, 0.7, dtype=torch.float)
            weight[labels == 1] = 0.3

            loss = torch.nn.functional.binary_cross_entropy(probabilities, labels, weight=weight)

            if config['training']['contrastive_regularizer']:
                embs1 = model.encoder(batch1)
                embs2 = model.encoder(batch2)
            
                # Расширяем до всех пар (batch_size ** 2)
                embs1_all = embs1.unsqueeze(1).expand(batch_size, batch_size, -1).reshape(-1, embs1.size(-1))
                embs2_all = embs2.unsqueeze(0).expand(batch_size, batch_size, -1).reshape(-1, embs2.size(-1))
            
                # Метки: 1 если i == j, иначе -1
                contrastive_loss_labels = 2 * labels - 1  # {0,1} → {-1,1}
            
                contrastive_loss = regularizer(embs1_all, embs2_all, contrastive_loss_labels)
                total_loss = loss + config['regularizer']['lambda'] * contrastive_loss
            else:
                total_loss = loss

            total_loss.backward()

            optimizer.step()
            train_loss += total_loss.item() * pairs_batch_size
            train_total += pairs_batch_size

            predicted_labels = (probabilities.detach().cpu().numpy() > 0.5).astype(int)
            true_labels = labels.cpu().numpy()

            update_metrics(train_metrics_sum, true_labels, predicted_labels)

        scheduler.step()

        train_metrics = compute_metrics(train_metrics_sum)
        train_loss /= train_total

        train_loss /= train_total

        # ---------------------
        # Валидация
        # ---------------------
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_metrics_sum = init_metrics()

        with torch.no_grad():
            for batch1, batch2 in val_loader:
                batch1, batch2 = batch1.to(device), batch2.to(device)
                batch_size = batch1.size(0)
                pairs_batch_size = batch_size ** 2

                probabilities = model(batch1, batch2).view(-1)

                labels = torch.eye(batch_size).to(device).view(-1)
                weight = torch.full_like(labels, 0.7, dtype=torch.float)
                weight[labels == 1] = 0.3

                loss = torch.nn.functional.binary_cross_entropy(probabilities, labels, weight=weight)

                if config['training']['contrastive_regularizer']:
                    embs1 = model.encoder(batch1)
                    embs2 = model.encoder(batch2)
                
                    # Расширяем до всех пар (batch_size ** 2)
                    embs1_all = embs1.unsqueeze(1).expand(batch_size, batch_size, -1).reshape(-1, embs1.size(-1))
                    embs2_all = embs2.unsqueeze(0).expand(batch_size, batch_size, -1).reshape(-1, embs2.size(-1))
                
                    # Метки: 1 если i == j, иначе -1
                    contrastive_loss_labels = 2 * labels - 1  # {0,1} → {-1,1}
                
                    contrastive_loss = regularizer(embs1_all, embs2_all, contrastive_loss_labels)
                    total_loss = loss + config['regularizer']['lambda'] * contrastive_loss
                else:
                    total_loss = loss

                val_loss += total_loss.item() * pairs_batch_size
                val_total += pairs_batch_size

                predicted_labels = (probabilities.detach().cpu().numpy() > 0.5).astype(int)
                true_labels = labels.cpu().numpy()
    
                update_metrics(val_metrics_sum, true_labels, predicted_labels)

        val_metrics = compute_metrics(val_metrics_sum)
        val_loss /= val_total

        # ---------------------
        # Логгирование
        # ---------------------
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_metrics["accuracy"]:.4f}, '
              f'Train F1: {train_metrics["f1"]:.4f}, '
              f'Train Recall: {train_metrics["recall"]:.4f}, '
              f'Train Precision: {train_metrics["precision"]:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_metrics["accuracy"]:.4f}, '
              f'Val F1: {val_metrics["f1"]:.4f}, '
              f'Val Recall: {val_metrics["recall"]:.4f}, '
              f'Val Precision: {val_metrics["precision"]:.4f}, '
              f'lr: {current_lr}')

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
        writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
        writer.add_scalar('Recall/Train', train_metrics['recall'], epoch)
        writer.add_scalar('Precision/Train', train_metrics['precision'], epoch)

        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
        writer.add_scalar('Recall/Val', val_metrics['recall'], epoch)
        writer.add_scalar('Precision/Val', val_metrics['precision'], epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, config)

    writer.close()
