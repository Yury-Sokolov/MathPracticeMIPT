#!/usr/bin/env python
import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from core import Simulation
from potential import MesonExchangePotential
from utils.config import create_arg_parser, load_config


def main():
    """
    Основная функция для запуска обучения UDE, генерации данных и анализа.
    """
    parser = create_arg_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        print("\nОшибка: Необходимо указать режим работы с аргументом --mode")
        return
    
    # Загружаем конфигурацию
    config = load_config(args)
    
    # Определяем устройство
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device_str}")
    
    # Запускаем соответствующий режим
    if args.mode == 'generate_data':
        from scripts.generate_data import generate_data
        generate_data(config)
        
    elif args.mode == 'train_ude':
        from scripts.train_ude import train_ude
        train_ude(config)
        
    elif args.mode == 'analyze_results':
        from scripts.analyze_results import analyze_results
        analyze_results(config)
        
    elif args.mode == 'run_simulation':
        from scripts.run_simulation import run_simulation
        run_simulation(config)
        
    else:
        print(f"Неизвестный режим: {args.mode}")
        parser.print_help()


if __name__ == "__main__":
    main()
