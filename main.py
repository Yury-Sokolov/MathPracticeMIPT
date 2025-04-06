import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

from core import Simulation
from potential import MesonExchangePotential

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    clustering_algorithm = DBSCAN(eps=1.5, min_samples=3)
    
    potential = MesonExchangePotential(
        g_att=13.5,
        g_rep=20.0,
        m_pi=0.70,
        m_rho=3.93,
        r_cutoff=5.0,
        r_core=0.3,
        device=device
    )

    sim = Simulation(
        potential,
        t_end=1.0,
        device=device,
        # adaptive_dt=True,
        dt_min=1e-16,
        tau_max=0.01,
        Imax=0.1,
        clustering_algorithm=clustering_algorithm
    )
    #
    # print("Запуск одиночной демонстрационной симуляции...")
    # sim.setup_impact_parameter(
    #     nucleus_count1=20,
    #     nucleus_count2=20,
    #     impact_parameter=1.0,
    #     relative_velocity=20.0,
    #     random_velocity=3.0,
    #     radius1=1.0,
    #     radius2=1.0
    # )
    
    # result = sim.run(save_interval=5, dt_initial=0.00001, max_steps=1000)
    #
    # print("Создание анимации столкновения...")
    # sim.create_animation(filename="plots/demo_collision.mp4", fps=30, limit=10)
    #
    #
    # print("Результат кластеризации доступен в переменной result['cluster_labels']")
    #
    # print("Визуализация результатов кластеризации...")
    # sim.cluster_analysis(clustering_algorithm, save_path="plots/demo_clusters.png", limit=10)
    #
    # print("\nЗапуск множественных столкновений для статистического анализа...")
    
    NUM_COLLISIONS = 100
    
    results = sim.run_multiple_collisions(
        count=NUM_COLLISIONS,
        nucleus_count1=20,
        nucleus_count2=20,
        velocity=20.0,
        max_impact_parameter=5.0,
        save_interval=10,
        dt_initial=0.001,
        max_steps=500,
        random_velocity=3.0,
        radius1=1.0,
        radius2=1.0
    )
    
    print("Анализ результатов множественных столкновений...")
    stats = Simulation.analyze_multiple_results(results)
    
    if stats is not None and 'figure' in stats:
        stats['figure'].savefig("plots/statistics.png", dpi=300)
        plt.close(stats['figure'])
        print("Статистика сохранена в plots/statistics.png")
    
    print("Эксперимент завершен!")
