import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from omegaconf import DictConfig
import logging
from tqdm import tqdm

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Добавляем корневую директорию в путь импорта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.simulation import Simulation
from potential.potential import MesonExchangePotential
from ude.ude_network import PotentialNN, KANPotentialModel
from utils.config import load_config, create_arg_parser
from utils.data_manager import DataManager, load_data, normalize_data
from utils.visualization import (
    plot_trajectory_comparison, 
    plot_force_comparison, 
    plot_potential_comparison, 
    create_collision_animation
)


def analyze_results(config: DictConfig) -> None:
    """
    Анализирует результаты обучения UDE модели.
    
    Args:
        config: Конфигурация с параметрами для анализа
    """
    logger.info("--- Запуск анализа результатов ---")
    start_time = time.time()
    
    # Проверяем и создаем необходимые директории
    output_dir = config.get('evaluation.analysis_output_dir', './analysis')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Результаты будут сохранены в {output_dir}")
    
    # Определяем устройство для вычислений
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    logger.info(f"Используется устройство: {device}")
    
    try:
        data_path = config.get('evaluation.data_file')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл с данными не найден: {data_path}")
            
        logger.info(f"Загрузка данных из {data_path}...")
        data = load_data(data_path)
        
        logger.info("Нормализация данных...")
        data, normalization_stats = normalize_data(data)
        
        logger.info(f"Данные успешно загружены. Размер: {len(data['positions'])}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return
    
    try:
        model_path = config.get('evaluation.model_load_path')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            
        model_type = config.get('model.type', 'potential_nn')
        hidden_dim = config.get('model.hidden_dim', 16)
        num_blocks = config.get('model.num_residual_blocks', 4)
        max_potential = config.get('model.max_potential', 50.0)
        dropout_rate = config.get('model.dropout_rate', 0.1)
        
        logger.info(f"Инициализация модели типа {model_type}...")
        
        if model_type == 'kan':
            logger.info("Инициализация KAN модели...")
            try:
                model = KANPotentialModel(
                    hidden_dim=hidden_dim,
                    num_layers=num_blocks,
                    max_potential=max_potential,
                    device=device
                )
            except ImportError as e:
                logger.warning(f"Ошибка: {e}")
                logger.warning("Использую PotentialNN вместо KAN.")
                model = PotentialNN(
                    hidden_dim=hidden_dim, 
                    num_blocks=num_blocks,
                    max_potential=max_potential,
                    dropout_rate=dropout_rate
                ).to(device)
        else:
            logger.info("Инициализация PotentialNN модели...")
            model = PotentialNN(
                hidden_dim=hidden_dim, 
                num_blocks=num_blocks,
                max_potential=max_potential,
                dropout_rate=dropout_rate
            ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info(f"Модель успешно загружена из {model_path}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return
    
    try:
        potential_params = config.get('potential', {})
        if all(key in potential_params for key in ['g_att', 'g_rep', 'm_pi', 'm_rho']):
            logger.info("Инициализация эталонного потенциала для сравнения...")
            true_potential = MesonExchangePotential(
                g_att=potential_params.get('g_att'),
                g_rep=potential_params.get('g_rep'),
                m_pi=potential_params.get('m_pi'),
                m_rho=potential_params.get('m_rho'),
                r_cutoff=potential_params.get('r_cutoff', 5.0),
                r_core=potential_params.get('r_core', 0.5),
                device=device
            )
        else:
            logger.warning("Не удалось инициализировать эталонный потенциал: недостаточно параметров")
            true_potential = None
    except Exception as e:
        logger.error(f"Ошибка при инициализации эталонного потенциала: {e}")
        true_potential = None
    
    logger.info("Анализ ошибок предсказания сил...")
    try:
        test_distances = torch.linspace(0.5, 5.0, 100, device=device)
        force_analysis_path = os.path.join(output_dir, "force_analysis.png")
        
        with torch.no_grad():
            test_positions = torch.zeros((100, 3), device=device)
            test_positions[:, 0] = test_distances 
            
            predicted_forces = model(test_positions)
            predicted_forces_np = predicted_forces.cpu().numpy()
            
            if true_potential:
                true_forces = true_potential.compute_force_only(test_positions)
                true_forces_np = true_forces.cpu().numpy()
                
                mse = mean_squared_error(true_forces_np[:, 0], predicted_forces_np[:, 0])
                mae = mean_absolute_error(true_forces_np[:, 0], predicted_forces_np[:, 0])
                
                logger.info(f"MSE между предсказанными и истинными силами: {mse:.6f}")
                logger.info(f"MAE между предсказанными и истинными силами: {mae:.6f}")
                
                plot_force_comparison(
                    distances=test_distances.cpu().numpy(),
                    predicted_forces=predicted_forces_np[:, 0],
                    true_forces=true_forces_np[:, 0],
                    save_path=force_analysis_path,
                    title="Сравнение предсказанных и истинных сил"
                )
                logger.info(f"График анализа сил сохранен в {force_analysis_path}")
            else:
                plt.figure(figsize=(10, 6))
                plt.plot(test_distances.cpu().numpy(), predicted_forces_np[:, 0], 'b-', label='Предсказанные силы')
                plt.xlabel('Расстояние')
                plt.ylabel('Сила')
                plt.title('Предсказанные силы от расстояния')
                plt.grid(True)
                plt.legend()
                plt.savefig(force_analysis_path)
                plt.close()
                logger.info(f"График предсказанных сил сохранен в {force_analysis_path}")
    except Exception as e:
        logger.error(f"Ошибка при анализе сил: {e}")
    
    logger.info("Анализ потенциальной энергии...")
    try:
        potential_analysis_path = os.path.join(output_dir, "potential_analysis.png")
        
        if hasattr(model, 'compute_potential'):
            with torch.no_grad():
                test_positions = torch.zeros((100, 3), device=device)
                test_positions[:, 0] = test_distances  # x-axis
                
                predicted_potential = model.compute_potential(test_positions)
                predicted_potential_np = predicted_potential.cpu().numpy()
                
                if true_potential and hasattr(true_potential, 'compute_potential'):
                    true_potential_values = true_potential.compute_potential(test_positions)
                    true_potential_np = true_potential_values.cpu().numpy()
                    
                    plot_potential_comparison(
                        distances=test_distances.cpu().numpy(),
                        predicted_potential=predicted_potential_np,
                        true_potential=true_potential_np,
                        save_path=potential_analysis_path,
                        title="Сравнение предсказанного и истинного потенциалов"
                    )
                    logger.info(f"График анализа потенциалов сохранен в {potential_analysis_path}")
                else:
                    plt.figure(figsize=(10, 6))
                    plt.plot(test_distances.cpu().numpy(), predicted_potential_np, 'r-', label='Предсказанный потенциал')
                    plt.xlabel('Расстояние')
                    plt.ylabel('Потенциальная энергия')
                    plt.title('Предсказанный потенциал от расстояния')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(potential_analysis_path)
                    plt.close()
                    logger.info(f"График предсказанного потенциала сохранен в {potential_analysis_path}")
        else:
            logger.warning("Модель не имеет метода compute_potential, анализ потенциальной энергии пропущен")
    except Exception as e:
        logger.error(f"Ошибка при анализе потенциальной энергии: {e}")
    
    if model_type == 'kan' and hasattr(model, 'get_formula'):
        try:
            logger.info("Извлечение символической формулы из KAN модели...")
            formula = model.get_formula(precision=4)
            
            formula_path = os.path.join(output_dir, "kan_formula.txt")
            with open(formula_path, 'w') as f:
                f.write(f"Символическая формула потенциала (KAN):\n\n")
                f.write(formula)
            
            logger.info(f"Символическая формула сохранена в {formula_path}")
            logger.info(f"Формула: {formula}")
        except Exception as e:
            logger.error(f"Ошибка при извлечении символической формулы: {e}")
    
    if 'trajectories' in data:
        try:
            logger.info("Анализ траекторий...")
            trajectory_analysis_path = os.path.join(output_dir, "trajectory_analysis.png")
            
            num_trajectories = min(5, len(data['trajectories']))
            selected_indices = np.random.choice(len(data['trajectories']), num_trajectories, replace=False)
            
            plot_trajectory_comparison(
                true_trajectories=[data['trajectories'][i] for i in selected_indices],
                predicted_trajectories=None, 
                save_path=trajectory_analysis_path,
                title="Визуализация траекторий"
            )
            logger.info(f"График анализа траекторий сохранен в {trajectory_analysis_path}")
        except Exception as e:
            logger.error(f"Ошибка при анализе траекторий: {e}")
    
    end_time = time.time()
    logger.info(f"--- Анализ результатов завершен за {end_time - start_time:.2f} секунд ---")


def analyze_potential_force(model, true_potential, output_dir):
    """
    Анализирует потенциал и силы, предсказанные моделью.
    
    Args:
        model: Обученная UDE модель
        true_potential: Истинный потенциал (если есть)
        output_dir: Директория для сохранения результатов
    """
    r_values = torch.linspace(0.2, 5.0, 100, device=model.scaling_factor.device)
    
    pos_vectors = torch.zeros((len(r_values), 3), device=model.scaling_factor.device)
    pos_vectors[:, 0] = r_values
    
    model.eval()
    with torch.no_grad():
        ude_potentials = model.compute_potential(pos_vectors)
        
        true_potentials = None
        if true_potential:
            try:
                true_potentials = true_potential.compute_potential(r_values)
            except:
                true_potentials = torch.zeros_like(ude_potentials)
                g_att = true_potential.g_att
                g_rep = true_potential.g_rep
                m_pi = true_potential.m_pi
                m_rho = true_potential.m_rho
                
                for i, r in enumerate(r_values):
                    if r > 0:
                        true_potentials[i] = g_rep * torch.exp(-m_rho*r) / r - g_att * torch.exp(-m_pi*r) / r
    
    pos_vectors_grad = pos_vectors.clone().requires_grad_(True)
    
    ude_potentials_grad = model.compute_potential(pos_vectors_grad)
    total_ude_potential = torch.sum(ude_potentials_grad)
    
    ude_forces = -torch.autograd.grad(
        total_ude_potential, pos_vectors_grad, 
        create_graph=False, retain_graph=False
    )[0]
    
    true_forces = None
    if true_potential:
        with torch.no_grad():
            try:
                true_forces = true_potential.compute_force(pos_vectors)
            except:
                true_forces = torch.zeros_like(ude_forces)
                g_att = true_potential.g_att
                g_rep = true_potential.g_rep
                m_pi = true_potential.m_pi
                m_rho = true_potential.m_rho
                
                term1 = g_rep * torch.exp(-m_rho*r_values) * (m_rho/r_values + 1/(r_values**2))
                term2 = g_att * torch.exp(-m_pi*r_values) * (m_pi/r_values + 1/(r_values**2))
                true_force_x = term1 - term2
                
                true_forces[:, 0] = true_force_x
    
    if true_forces is not None and true_potentials is not None:
        pot_error = torch.mean((ude_potentials - true_potentials) ** 2).item()
        force_error = torch.mean((ude_forces - true_forces) ** 2).item()
        
        max_true_pot = torch.max(torch.abs(true_potentials)).item()
        max_ude_pot = torch.max(torch.abs(ude_potentials)).item()
        
        max_true_force = torch.max(torch.abs(true_forces)).item()
        max_ude_force = torch.max(torch.abs(ude_forces)).item()
        
        print(f"Ошибка потенциала: {pot_error:.6f}")
        print(f"Ошибка силы: {force_error:.6f}")
        print(f"Макс. истинный потенциал: {max_true_pot:.6f}, Макс. UDE потенциал: {max_ude_pot:.6f}")
        print(f"Макс. истинная сила: {max_true_force:.6f}, Макс. UDE сила: {max_ude_force:.6f}")
    
    if true_forces is not None:
        plot_force_comparison(
            r_values, 
            ude_forces, 
            true_forces, 
            save_path=os.path.join(output_dir, "force_comparison.png"),
            title="Сравнение истинных и предсказанных UDE сил"
        )
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(r_values.cpu().numpy(), ude_forces[:, 0].cpu().numpy(), 'r-', label='UDE Force')
        plt.xlabel("Distance (r)")
        plt.ylabel("Force magnitude (x-component)")
        plt.title("UDE предсказанные силы")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "ude_forces.png"))
        plt.close()
    
    if true_potentials is not None:
        plot_potential_comparison(
            r_values, 
            ude_potentials, 
            true_potentials, 
            save_path=os.path.join(output_dir, "potential_comparison.png"),
            title="Сравнение истинного и предсказанного UDE потенциала"
        )
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(r_values.cpu().numpy(), ude_potentials.cpu().numpy(), 'g-', label='UDE Potential')
        plt.xlabel("Distance (r)")
        plt.ylabel("Potential")
        plt.title("UDE предсказанный потенциал")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "ude_potential.png"))
        plt.close()
    
    if hasattr(model, 'get_symbolic_formula'):
        try:
            print("\n--- Символическая формула потенциала ---")
            kan_formula = model.get_symbolic_formula(precision=4, simplify=True)
            print(f"\n{kan_formula}\n")
            
            formula_file = os.path.join(output_dir, "kan_formula.txt")
            with open(formula_file, 'w') as f:
                f.write(f"KAN формула потенциала:\n{kan_formula}\n")
            
            print(f"Формула сохранена в {formula_file}")
        except Exception as e:
            print(f"Ошибка при извлечении символической формулы: {e}")


def analyze_trajectories(model, true_sim, ude_sim, data_manager, config, output_dir):
    """
    Анализирует траектории для разных потенциалов.
    
    Args:
        model: Обученная UDE модель
        true_sim: Симуляция с истинным потенциалом
        ude_sim: Симуляция с UDE моделью
        data_manager: Менеджер данных
        config: Конфигурация
        output_dir: Директория для сохранения результатов
    """
    setup_params = data_manager.data.get('setup_base_params', {})
    
    n_impact_params = 3
    max_impact_parameter = setup_params.get('impact_parameter', 3.0)
    impact_params = torch.linspace(0.1, max_impact_parameter, n_impact_params)
    
    prediction_steps = config.get('evaluation.prediction_steps', 500)
    dt_initial = data_manager.data.get('run_params', {}).get('dt_initial', 0.001)
    
    for i, b in enumerate(impact_params):
        print(f"\nАнализ траектории для параметра удара b = {b:.2f}...")
        
        if true_sim:
            true_sim.setup_impact_parameter(
                impact_parameter=b.item(),
                nucleus_count1=setup_params.get('nucleus_count1', 2),
                nucleus_count2=setup_params.get('nucleus_count2', 2),
                relative_velocity=setup_params.get('relative_velocity', 1.0),
                random_velocity=setup_params.get('random_velocity', 0.1),
                radius1=setup_params.get('radius1', 1.0),
                radius2=setup_params.get('radius2', 1.0)
            )
            
            true_result = true_sim.run(
                save_interval=1,
                dt_initial=dt_initial,
                max_steps=prediction_steps,
                hide_progress=False
            )
            
            true_positions = torch.tensor(true_result['positions'], device=model.scaling_factor.device)
            true_times = torch.tensor(true_result['times'], device=model.scaling_factor.device)
        else:
            true_positions = None
            true_times = None
        
        ude_sim.setup_impact_parameter(
            impact_parameter=b.item(),
            nucleus_count1=setup_params.get('nucleus_count1', 2),
            nucleus_count2=setup_params.get('nucleus_count2', 2),
            relative_velocity=setup_params.get('relative_velocity', 1.0),
            random_velocity=setup_params.get('random_velocity', 0.1),
            radius1=setup_params.get('radius1', 1.0),
            radius2=setup_params.get('radius2', 1.0)
        )
        
        ude_result = ude_sim.run(
            save_interval=1,
            dt_initial=dt_initial,
            max_steps=prediction_steps,
            hide_progress=False
        )
        
        ude_positions = torch.tensor(ude_result['positions'], device=model.scaling_factor.device)
        ude_times = torch.tensor(ude_result['times'], device=model.scaling_factor.device)
        
        compare_times = ude_times if true_times is None else true_times
        
        if true_positions is not None and len(true_times) != len(ude_times):
            print(f"Интерполируем траектории для корректного сравнения...")
            
            min_time_len = min(len(true_times), len(ude_times))
            compare_times = compare_times[:min_time_len]
            
            if len(ude_times) > min_time_len:
                ude_positions = ude_positions[:min_time_len]
            if len(true_times) > min_time_len:
                true_positions = true_positions[:min_time_len]
        
        plot_path = os.path.join(output_dir, f"trajectory_comparison_b{b:.2f}.png")
        plot_trajectory_comparison(
            ude_positions,
            true_positions if true_positions is not None else ude_positions,
            noisy_positions=None,
            times=compare_times,
            num_particles=ude_positions.shape[1],
            save_path=plot_path,
            title=f"Сравнение траекторий (b={b:.2f})"
        )
        
        anim_path = os.path.join(output_dir, f"collision_animation_b{b:.2f}.mp4")
        create_collision_animation(
            ude_positions,
            compare_times,
            anim_path,
            fps=30,
            limit=10.0,
            title=f"UDE Столкновение (b={b:.2f})"
        )
        
        if true_positions is not None:
            if 'energies' in true_result and 'energies' in ude_result:
                true_energies = true_result['energies']
                ude_energies = ude_result['energies']
                
                min_len = min(len(true_energies['kinetic']), len(ude_energies['kinetic']))
                
                plt.figure(figsize=(14, 7))
                plt.subplot(1, 2, 1)
                plt.plot(true_energies['kinetic'][:min_len], label='True Kinetic')
                plt.plot(ude_energies['kinetic'][:min_len], label='UDE Kinetic')
                plt.title("Сравнение кинетической энергии")
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(true_energies['potential'][:min_len], label='True Potential')
                plt.plot(ude_energies['potential'][:min_len], label='UDE Potential')
                plt.title("Сравнение потенциальной энергии")
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"energy_comparison_b{b:.2f}.png"))
                plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = create_arg_parser(mode='analyze_results')
    args = parser.parse_args()
    
    try:
        config = load_config(args)
        analyze_results(config)
    except Exception as e:
        logger.error(f"Ошибка при выполнении анализа: {e}")
        sys.exit(1) 