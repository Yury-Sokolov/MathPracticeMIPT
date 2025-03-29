import numpy as np
from sklearn.cluster import DBSCAN

from core import Cluster, Simulation
from potential import YukawaPotential

if __name__ == "__main__":
    potential = YukawaPotential(V0=-10.0, alpha=1.0, r1=3.0)
    # potential.plot()
    sim = Simulation(potential, epsilon=1e-8, t_end=2.)

    cluster1 = Cluster(position=np.array([-5.0, 0.0, 0.0]),
                       velocity=np.array([5., 0., 0.]),
                       random_velocity=3,
                       radius=1.5,
                       N=20)

    cluster2 = Cluster(position=np.array([5.0, 0.0, 0.0]),
                       velocity=np.array([-5., 0., 0.]),
                       random_velocity=3,
                       radius=1.5,
                       N=20)

    sim.add_cluster(cluster1)
    sim.add_cluster(cluster2)

    sim.run(save_interval=10, dt=0.0001)

    sim.create_animation(filename="plots/result.mp4", fps=96, limit=10)

    clustering = DBSCAN(eps=1.5, min_samples=3)
    sim.cluster_analysis(clustering, save_path="plots/result.png",  limit=10)
