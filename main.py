import numpy as np
from sklearn.cluster import DBSCAN

from core import Cluster, Simulation
from potential import YukawaPotential

if __name__ == "__main__":
    potential = YukawaPotential(V0=10.0, alpha=1.0, r1=3.0)

    sim = Simulation(potential, t_end=1., convergence_threshold=1e-2, delta_time=60)

    cluster1 = Cluster(position=np.array([-5.0, 0.0, 0.0]),
                       velocity=np.array([1, 0.0, 0.0]),
                       radius=0.5,
                       N=50)

    cluster2 = Cluster(position=np.array([5.0, 0.0, 0.0]),
                       velocity=np.array([-1, 0.0, 0.0]),
                       radius=0.5,
                       N=50)

    sim.add_cluster(cluster1)
    sim.add_cluster(cluster2)

    sim.run(save_interval=0.0001)

    sim.create_animation(filename="plots/collision.mp4", fps=24)

    clustering = DBSCAN(eps=1.5, min_samples=3)
    sim.cluster_analysis(clustering, save_path="plots/clusters1.png")
