import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import numpy as np
import wandb
from omegaconf import DictConfig
from torch.cuda.amp import autocast, GradScaler

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from ude.ude_network import PotentialNN, KANPotentialModel
from ude.loss_manager import LossManager
from ude.trainer import Trainer
from utils.data_manager import load_data, normalize_data, prepare_training_data
from utils.visualization import plot_loss_curves, plot_potential_force_curves, plot_force_comparison
from potential.potential import MesonExchangePotential


def train_ude(config: DictConfig) -> None:
    """
    Функция для обучения модели UDE с заданными параметрами.
    
    Args:
        config: Конфигурация с параметрами обучения
    """
    print("\n--- Запуск обучения UDE ---")
    start_time = time.time()
    
    output_dir = config.get('training.output_dir', './models')
    os.makedirs(output_dir, exist_ok=True)
    
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Используется устройство: {device}")
    
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_wandb = config.get('wandb.use', False)
    if use_wandb:
        try:
            wandb.init(
                project=config.get('wandb.project', 'physics-ude'),
                entity=config.get('wandb.entity', None),
                tags=config.get('wandb.tags', None),
                config=config
            )
            print("Weights & Biases инициализирован.")
        except Exception as e:
            print(f"Ошибка инициализации Weights & Biases: {e}")
            use_wandb = False
    
    try:
        data_path = config.get('data.train_file', './data/ude_data.pt')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл с данными не найден: {data_path}")
            
        print(f"Загрузка данных из {data_path}...")
        data = load_data(data_path)
        
        print("Нормализация данных...")
        data, normalization_stats = normalize_data(data)
        
        print("Подготовка данных для обучения...")
        train_data, val_data, test_data = prepare_training_data(
            data, 
            train_ratio=config.get('data.train_ratio', 0.7),
            val_ratio=config.get('data.val_ratio', 0.15),
            test_ratio=config.get('data.test_ratio', 0.15)
        )
        
        batch_size = config.get('training.batch_size', 128)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        print(f"Подготовка данных завершена. Размеры: обучение({len(train_data)}), валидация({len(val_data)}), тест({len(test_data)})")
    except Exception as e:
        print(f"Ошибка при подготовке данных: {e}")
        if use_wandb:
            wandb.finish()
        return
    
    try:
        model_type = config.get('model.type', 'potential_nn')
        hidden_dim = config.get('model.hidden_dim', 16)
        num_blocks = config.get('model.num_residual_blocks', 4)
        max_potential = config.get('model.max_potential', 50.0)
        dropout_rate = config.get('model.dropout_rate', 0.1)
        
        print(f"Инициализация модели типа {model_type}...")
        
        if model_type == 'kan':
            print("Инициализация KAN модели...")
            try:
                model = KANPotentialModel(
                    hidden_dim=hidden_dim,
                    num_layers=num_blocks,
                    max_potential=max_potential,
                    device=device
                )
            except ImportError as e:
                print(f"Ошибка: {e}")
                print("Использую PotentialNN вместо KAN.")
                model = PotentialNN(
                    hidden_dim=hidden_dim, 
                    num_blocks=num_blocks,
                    max_potential=max_potential,
                    dropout_rate=dropout_rate
                ).to(device)
        else:
            print("Инициализация PotentialNN модели...")
            model = PotentialNN(
                hidden_dim=hidden_dim, 
                num_blocks=num_blocks,
                max_potential=max_potential,
                dropout_rate=dropout_rate
            ).to(device)
        
        print(f"Модель создана. Параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    except Exception as e:
        print(f"Ошибка при инициализации модели: {e}")
        if use_wandb:
            wandb.finish()
        return
    
    try:
        mse_weight = config.get('training.mse_weight', 1.0)
        directional_weight = config.get('training.directional_weight', 1.0)
        consistency_weight = config.get('training.consistency_weight', 0.5)
        smoothness_weight = config.get('training.smoothness_weight', 0.01)
        
        loss_manager = LossManager(
            mse_weight=mse_weight,
            directional_weight=directional_weight,
            consistency_weight=consistency_weight,
            smoothness_weight=smoothness_weight,
            device=device
        )
        
        print(f"Loss Manager инициализирован с весами: MSE({mse_weight}), Dir({directional_weight}), Consist({consistency_weight}), Smooth({smoothness_weight})")
    except Exception as e:
        print(f"Ошибка при инициализации Loss Manager: {e}")
        if use_wandb:
            wandb.finish()
        return
    
    try:
        lr = config.get('training.learning_rate', 0.001)
        optimizer_type = config.get('training.optimizer', 'adam')
        
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        elif optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        else:
            print(f"Неизвестный тип оптимизатора: {optimizer_type}. Используем Adam.")
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        scheduler_type = config.get('training.scheduler', 'one_cycle')
        epochs = config.get('training.epochs', 100)
        
        if scheduler_type.lower() == 'one_cycle':
            steps_per_epoch = len(train_loader)
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=lr,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif scheduler_type.lower() == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=True
            )
        else:
            print(f"Неизвестный тип планировщика: {scheduler_type}. Планировщик не будет использоваться.")
            scheduler = None
            
        print(f"Оптимизатор: {optimizer_type}, LR: {lr}, Планировщик: {scheduler_type}")
    except Exception as e:
        print(f"Ошибка при инициализации оптимизатора: {e}")
        if use_wandb:
            wandb.finish()
        return
    
    true_potential = None
    try:
        if config.get('training.use_reference_potential', False):
            true_potential = MesonExchangePotential(device=device)
            print("Эталонный потенциал инициализирован.")
    except Exception as e:
        print(f"Ошибка при инициализации эталонного потенциала: {e}")
        true_potential = None
    
    use_amp = config.get('training.use_amp', False) and device.type == 'cuda'
    if use_amp:
        scaler = GradScaler()
        print("Включен режим смешанной точности (AMP).")
    else:
        scaler = None
    
    try:
        model_save_path = os.path.join(output_dir, config.get('training.model_save_name', 'ude_model.pt'))
        
        trainer = Trainer(
            model=model,
            loss_manager=loss_manager,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_save_path=model_save_path,
            config=config,
            true_potential=true_potential,
            use_wandb=use_wandb
        )
        
        print(f"Trainer инициализирован. Модель будет сохранена в {model_save_path}")
    except Exception as e:
        print(f"Ошибка при инициализации Trainer: {e}")
        if use_wandb:
            wandb.finish()
        return
    
    try:
        print(f"\nНачало обучения на {epochs} эпохах...")
        
        if use_amp:
            trainer.train_with_amp(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                early_stopping_patience=config.get('training.early_stopping_patience', 20),
                scaler=scaler
            )
        else:
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                early_stopping_patience=config.get('training.early_stopping_patience', 20)
            )
        
        test_loss = trainer.evaluate(test_loader)
        print(f"\nОценка на тестовом наборе: Total Loss = {test_loss['total']:.4f}")
        
        plot_loss_curves(
            train_losses=trainer.train_losses,
            val_losses=trainer.val_losses,
            save_path=os.path.join(output_dir, 'loss_curves.png')
        )
        
        if true_potential:
            plot_potential_force_curves(
                ude_model=model,
                true_potential=true_potential,
                save_path=os.path.join(output_dir, 'potential_force_curves.png')
            )
        
        end_time = time.time()
        print(f"\n--- Обучение UDE завершено за {end_time - start_time:.2f} секунд ---")
        print(f"Модель сохранена в {model_save_path}")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
    finally:
        if use_wandb:
            wandb.finish()


def train_with_amp(trainer, train_loader, val_loader, epochs, early_stopping_patience, scaler):
    """
    Реализация обучения с использованием смешанной точности.
    
    Args:
        trainer: Объект Trainer
        train_loader: Загрузчик обучающих данных
        val_loader: Загрузчик данных валидации
        epochs: Количество эпох обучения
        early_stopping_patience: Терпение для ранней остановки
        scaler: Объект GradScaler для масштабирования градиентов в смешанной точности
    """
    trainer.model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        trainer.model.train()
        total_losses = {}
        num_batches = len(train_loader)
        
        for batch_idx, (rel_pos, accelerations, *additional_data) in enumerate(train_loader):
            rel_pos = rel_pos.to(trainer.device)
            accelerations = accelerations.to(trainer.device)
            
            if len(additional_data) >= 2:
                distances = additional_data[0].to(trainer.device)
                directions = additional_data[1].to(trainer.device)
            else:
                distances = None
                directions = None
            
            trainer.optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                predicted_forces = trainer.model(rel_pos)
                
                losses = trainer.loss_manager.compute_losses(
                    rel_pos=rel_pos,
                    predicted_forces=predicted_forces,
                    target_accelerations=accelerations,
                    distances=distances,
                    directions=directions
                )
                
                total_loss = losses['total']
            
            scaler.scale(total_loss).backward()
            
            scaler.step(trainer.optimizer)
            scaler.update()
            
            if isinstance(trainer.scheduler, OneCycleLR):
                trainer.scheduler.step()
            
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        for k in total_losses:
            total_losses[k] /= num_batches
        
        trainer.train_losses.append(total_losses)
        
        val_losses = trainer.evaluate(val_loader)
        trainer.val_losses.append(val_losses)
        
        if isinstance(trainer.scheduler, ReduceLROnPlateau):
            trainer.scheduler.step(val_losses['total'])
        
        epoch_time = time.time() - epoch_start_time
        print(f"Эпоха {epoch+1}/{epochs} | Время: {epoch_time:.2f}с | Потери: Обучение({total_losses['total']:.4f}) Валидация({val_losses['total']:.4f})")
        
        if trainer.use_wandb:
            wandb_log = {
                'epoch': epoch,
                'train_loss': total_losses['total'],
                'val_loss': val_losses['total'],
                'learning_rate': trainer.optimizer.param_groups[0]['lr'],
            }
            
            for k, v in total_losses.items():
                if k != 'total':
                    wandb_log[f'train_{k}'] = v
            
            for k, v in val_losses.items():
                if k != 'total':
                    wandb_log[f'val_{k}'] = v
            
            wandb.log(wandb_log)
        
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            trainer.save_model()
            patience_counter = 0
            print(f"Эпоха {epoch+1}: Новая лучшая модель сохранена!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Ранняя остановка на эпохе {epoch+1}")
                break
    
    trainer.load_model()
    print(f"Загружена лучшая модель с валидационной потерей {best_val_loss:.4f}")


Trainer.train_with_amp = train_with_amp


if __name__ == "__main__":
    import argparse
    from utils.config import create_arg_parser, load_config
    
    parser = create_arg_parser(mode='train_ude')
    args = parser.parse_args()
    
    config = load_config(args)
    train_ude(config) 