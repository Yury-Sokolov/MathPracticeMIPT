import torch
from sklearn.cluster import DBSCAN

from core import Cluster, Simulation
from potential import ModifiedYukawaPotential

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    potential = ModifiedYukawaPotential(
        V0=-10.0,
        alpha=1.0,
        r1=3.0,
        V_rep=-20.0,
        beta=5.0,
        r_core=0.3,
        device=device
    )
    sim = Simulation(potential, dt_min=1e-16, t_end=1., device=device)

    cluster1 = Cluster(position=torch.tensor([-4.5, 0.0, 0.0]),
                       velocity=torch.tensor([10., 0., 0.]),
                       random_velocity=3,
                       radius=1,
                       N=20)

    cluster2 = Cluster(position=torch.tensor([4.5, 0.0, 0.0]),
                       velocity=torch.tensor([-10., 0., 0.]),
                       random_velocity=3,
                       radius=1,
                       N=20)

    sim.add_cluster(cluster1)
    sim.add_cluster(cluster2)


    sim.run(save_interval=1, dt_initial=0.00001, max_steps=2)



    sim.create_animation(filename="plots/result.mp4", fps=30, limit=10)

    clustering = DBSCAN(eps=1.5, min_samples=3)
    sim.cluster_analysis(clustering, save_path="plots/result.png",  limit=10)
