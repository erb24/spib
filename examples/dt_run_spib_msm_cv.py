# import the required packages
import numpy as np
import torch
import os
import torch.nn.functional as F
import random

from spib.spib import SPIB
from spib.utils import prepare_data, DataNormalize

from sklearn.model_selection import KFold
from msmbuilder.msm import MarkovStateModel
from msmbuilder.cluster import KMeans
from msmbuilder.decomposition import tICA
from sklearn.cluster import MiniBatchKMeans


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")

# Settings
# ------------------------------------------------------------------------------

# Model parameters
# Time delay delta t in terms of # of minimal time resolution of the trajectory data
dt_list = [1, 10, 100, 1000, 10000] #[1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

# Subsampling timestep
subsampling_timestep = 1

# Dimension of RC or bottleneck
z_dim  = 2

# Hyper-parameter beta
beta_list = [1e-2, 1e-1]

# Training parameters
batch_size = 4096

# tolerance of loss change for measuring the convergence of the training
tol = 0.002

# Number of epochs with the change of the state population smaller than the threshold after which this iteration of the training finishes
patience = 1

# Minimum refinements
refinements = 15

# Other controls

nn1 = 64
nn2 = 64
nn_list = [(16, 16), (32, 32), (64, 64)]

lr = 1e-4
gamma = 0.0
penalty = "None"

# Import data
traj_data_list = []
for i in range(num_div):
	traj_data_list.append(np.load('restart%s/sparse_calaculated_colvar.npy'))

# Generate initial state labels using tICA + k-means
tica_n_components = 2
tica_lag_time = 1000
kmeans_clusters = 10

## get tICA projection
## kinetic_mapping reweigts the eigenvectors by the corresponding eigenvalues, J. Chem. Theory Comput. 2015, 11, 10, 5002–5011
tica = tICA(n_components=tica_n_components, lag_time=tica_lag_time, kinetic_mapping=True)
tica.fit(traj_data_list)

tica_trajs = tica.transform(traj_data_list)
## do clustering on the tICA space as initial labels for SPIB
cluster = KMeans(n_clusters=kmeans_clusters, random_state=0, n_init=5)
cluster.fit(tica_trajs)
cluster_data = cluster.transform(tica_trajs)

initial_label = cluster_data[0]

# data normalization
data_transform = DataNormalize(mean=traj_data.mean(axis=0), std=traj_data.std(axis=0))

# data shape
data_shape = traj_data.shape[1:]
output_dim = np.max(initial_label) + 1

# Random seed
seed = 0

# Set random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

### load all the trajectories into a list
num_div = 10
divtraj = []
divlabel = []
for i in range(num_div):
	divtraj.append(np.load('restart%s/sparse_calaculated_colvar.npy'))
	divlabel.append(np.load('restart%s/sparse_init_traj_labels.npy'))

# split data into train and test set
indices = list(range(num_div))
split = int(np.floor(0.1 * num_div))

np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]



for beta in beta_list:
	for nn in nn_list:
		for dt in dt_list:

			nn1 = nn[0]
			nn2 = nn[1]

			# By default, we save all the results in subdirectories of the following path.
			base_path = "nonlinear_SPIB_dt_%d" % (dt)

			IB_path = os.path.join(base_path, "nn%d_nn%d_spib/" % (nn1, nn2))

			# Set random seed
			random.seed(seed)
			np.random.seed(seed)
			torch.manual_seed(seed)

			train_dataset, test_dataset = prepare_data(divtraj, divlabel, weight_list=None, output_dim=output_dim,
													lagtime=dt, subsampling_timestep=subsampling_timestep, 
													train_indices=train_indices, test_indices=test_indices, device=device,
													Ut_list=None)

			# scoring model for prepared dataset
			# lagtime should be divided by subsampling_timestep 

			IB = SPIB(output_dim=output_dim, data_shape=data_shape, encoder_type='Nonlinear', z_dim=z_dim,  lagtime=dt,
					beta=beta, learning_rate=lr, lr_scheduler_gamma=1, device=device,
					path=IB_path, UpdateLabel=True, neuron_num1=nn1, neuron_num2=nn2, data_transform=data_transform)

			IB.to(device)

			IB.fit(train_dataset, test_dataset, batch_size=batch_size, tolerance=tol, patience=patience, refinements=refinements, mask_threshold=0, index=seed)

			# save torch model
			torch.save(IB, IB.output_path + '_final_SPIB%i.model' % seed)

			# get transformed trajectory
			train_SPIB_labels = []
			train_SPIB_prediction = []
			train_SPIB_z_latent = []
			for i in train_indices:
				labels, prediction, z_latent, _ = IB.transform(divtraj[i], batch_size=batch_size, to_numpy=True)
				train_SPIB_labels += [labels]
				train_SPIB_prediction += [prediction]
				train_SPIB_z_latent += [z_latent]

			test_SPIB_labels = []
			test_SPIB_prediction = []
			test_SPIB_z_latent = []
			for i in test_indices:
				labels, prediction, z_latent, _ = IB.transform(divtraj[i], batch_size=batch_size, to_numpy=True)
				test_SPIB_labels += [labels]
				test_SPIB_prediction += [prediction]
				test_SPIB_z_latent += [z_latent]
