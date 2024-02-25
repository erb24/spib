import torch
import numpy as np
import jacobian
import util

def mean(z, scaled = True):
	if scaled:
		z = (z - z.min()) / (z.max() - z.min())

	if z.shape[1] > 1:
		mean_z = torch.mean(z, dim = 0)
		aux_loss = torch.abs(mean_z[1] - mean_z[0])
	# one-dimensional case
	else:
		mean_z = torch.mean(z)
		aux_loss = torch.abs(mean_z)

	return aux_loss


def var(z, scaled = True):
	if scaled:
		z = (z - z.min()) / (z.max() - z.min())

	if z.shape[1] > 1:
		mean_z = torch.mean(z, dim = 0)
		aux_loss = ((torch.mean(z[:,0]**2) - mean_z[0]**2) / torch.sum(torch.mean(z * z, dim = 0) - mean_z**2))
	# one-dimensional case
	else:
		mean_z = torch.mean(z)
		aux_loss = ((torch.mean(z**2) - mean_z**2))

	return aux_loss

# penalize by the information entropy of the latent variable -- no Jacobian, so not exact, but fine for a first try

# define the kernel density estimator implemented in PyTorch
# code adapted from:
# https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py

def kde(z, bandwidth = 0.1, bins = 50):
	if type(z) == torch.Tensor:
		pass
	else:
		z = torch.Tensor(z)
	samples = torch.linspace(float(z.min()), float(z.max()), bins)[:, None]
	samples = samples.view(samples.shape[0], 1, *samples.shape[1:])
	z = z.view(1, z.shape[0], *z.shape[1:])

	dz = samples - z
	dims = tuple(range(len(dz.shape))[2:])
	var = bandwidth**2
	exp = torch.exp(-torch.norm(dz, p = 2, dim = dims)**2 / (2 * var))
	coef = 1 / torch.sqrt(torch.tensor(2 * np.pi * var))
	return (coef * exp).mean(dim = 1), samples

def Hz(z, jac = None, weights = None, bins = 50, bandwidth = 0.1):
	if z.shape[1] == 2:
		hist1, z_sampled1 = jacobian.kde_jacobian(z[:,0][:, None], jac = jac, bandwidth = bandwidth, bins = bins, weights = weights)
		hist2, z_sampled2 = jacobian.kde_jacobian(z[:,1][:, None], jac = jac, bandwidth = bandwidth, bins = bins, weights = weights)
		p_z1 = hist1 
		p_z2 = hist2 
		dummy1 = p_z1 * torch.log(p_z1)
		dummy1[torch.isnan(dummy1)] = 0.0
		dummy1[torch.isinf(abs(dummy1))] = 0.0
		dummy2 = p_z2 * torch.log(p_z2)
		dummy2[torch.isnan(dummy2)] = 0.0
		dummy2[torch.isinf(abs(dummy2))] = 0.0
		H_z1 = -torch.nansum(dummy1)
		H_z2 = -torch.nansum(dummy2)
		aux_loss = torch.abs(H_z1 - H_z2)
	elif z.shape[1] == 1:
		hist1, z_sampled1 = jacobian.kde_jacobian(z, jac = jac, bandwidth = bandwidth, bins = bins, weights = weights)
		p_z = hist1 
		dummy1 = p_z1 * torch.log(p_z1)
		dummy1[torch.isnan(dummy1)] = 0.0
		dummy1[torch.isinf(abs(dummy1))] = 0.0
		H_z = -torch.nansum(dummy1)
		aux_loss = H_z
	else:
		Hz_sum = 0
		for i in range(z.shape[1]):
			if i == 0:
				hist1, z_sampled1 = jacobian.kde_jacobian(z[:,i][:, None], jac = jac, bandwidth = bandwidth, bins = bins, weights = weights)
				p_z1 = hist1 
				dummy1 = p_z1 * torch.log(p_z1)
				dummy1[torch.isnan(dummy1)] = 0.0
				dummy1[torch.isinf(abs(dummy1))] = 0.0
				H_z1 = -torch.nansum(dummy1)
				Hz_sum += H_z1
			else:
				hist1, z_sampled1 = jacobian.kde_jacobian(z[:,0][:, None], jac = jac, bandwidth = bandwidth, bins = bins, weights = weights)
				p_z = hist 
				dummy1 = p_z * torch.log(p_z)
				dummy1[torch.isnan(dummy1)] = 0.0
				dummy1[torch.isinf(abs(dummy1))] = 0.0
				H_z = -torch.nansum(dummy1)
				Hz_sum += H_z
		aux_loss = H_z1 / Hz_sum
		#raise NotImplementedError('This penalty method is not supported for latent variable spaces of dimensionality greater than 2.')
	return aux_loss

def dSz(data, z, Ut, bins = 50, bandwidth = 0.01, output_thermo = False):
	z, dzdx_list, dzdy_list = jacobian.torch_jacobian_1d(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')

	if z.shape[1] == 1:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		G1 = G1[nan_filter]
		ref = G1.argmin()
		G11 = G1 - G1[ref]

		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		Uz = Uz[nan_filter]
		U11 = Uz - Uz[ref]
		S11 = (G11 - U11)
		aux_loss = S11.max() - S11[ref] #S11.min()
		
	elif z.shape[1] == 2:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		G1 = G1[nan_filter]
		ref = G1.argmin()
		G11 = G1 - G1[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		Uz = Uz[nan_filter]
		U11 = Uz - Uz[ref]
		S11 = (G11 - U11)
		
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[1], dzdy_list[1]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		G2 = G2[nan_filter]
		ref = G2.argmin()
		G22 = G2 - G2[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		Uz = Uz[nan_filter]
		U22 = Uz - Uz[ref]
		S22 = (G22 - U22)
		aux_loss = torch.abs(S11.max() - S11[ref] - S22.max() + S22[ref])
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(torch.column_stack([dzdx_list[i], dzdy_list[i]]), dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))

	if output_thermo:
		return aux_loss, G11, U11, S11
	else:
		return aux_loss

def dSz_path(data, z, Ut, labels, s1 = 0, s2 = 9, bins = 50, bandwidth = 0.05, output_thermo = False):
	# normalize z
	#z = (z - z.min()) / (z.max() - z.min())
	z, dzdx_list, dzdy_list = jacobian.torch_jacobian_1d(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		#G1 = G1[nan_filter]
		try:
			ref = G1[nan_filter].min()
		except RuntimeError:
			ref = G1.min()

		G11 = G1 - ref

		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		#Uz = Uz[nan_filter]
		try:
			U11 = Uz - Uz[G1 == ref]
		except RuntimeError:
			U11 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S11 = (G11 - U11)

		# pick out the path to 'inflate'
		dz = 1. / bins #torch.abs((zbins[1] - zbins[0]).ravel())
		labels = torch.argmax(labels, axis = 1)

		# select endpoints plus a buffer of one standard deviation
		z1 = z[:,0][labels == s1].mean()
		z1_std = z[:,0][labels == s1].std()
		z2 = z[:,0][labels == s2].mean()
		z2_std = z[:,0][labels == s2].std()

		if (len(z[:,0][labels == s1]) == 0) or (len(z[:,0][labels == s2]) == 0):
			path = (zbins.ravel() == zbins.ravel())
		else:
			# select the path by excluding the endpoints
			if z1 < z2:
				path = ~((zbins >= float(z1 + z1_std)) ^ (zbins <= float(z2 - z2_std)))
			else:
				path = ~((zbins >= float(z2 + z2_std)) ^ (zbins <= float(z1 - z1_std)))
			path = torch.ravel(path)

		#with open('debug.txt', "w+") as f:
		#	f.write(str(S11) + '\n')
		#	f.write(str(path) + '\n')


		# find the total entropy along the path between endpoints to 'inflate'
		path_sum = (dz * S11[path]).sum()
		aux_loss =  path_sum

	# right now, the code has not been changed for the cases where the latent space is
	# more than one-dimensional
	elif z.shape[1] == 2:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		G1 = G1[nan_filter]
		ref = G1.argmin()
		G11 = G1 - G1[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		Uz = Uz[nan_filter]
		U11 = Uz - Uz[ref]
		S11 = (G11 - U11)
		
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[1], dzdy_list[1]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		G2 = G2[nan_filter]
		ref = G2.argmin()
		G22 = G2 - G2[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		Uz = Uz[nan_filter]
		U22 = Uz - Uz[ref]
		S22 = (G22 - U22)
		aux_loss = torch.abs(S11.max() - S11[ref] - S22.max() + S22[ref])
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(torch.column_stack([dzdx_list[i], dzdy_list[i]]), dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		print(G11, U11, S11)
		return aux_loss, G11, U11, S11
	else:
		return aux_loss

def dSz_point(data, z, Ut, bins = 50, bandwidth = 1.0, output_thermo = False, b = 1.0, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0 :
				S11_filtered = S11[path][nan_filter]
				path_sum = torch.abs(G11 + (S11_filtered.max() - S11_filtered.min()))
				aux_loss =  -path_sum
			else:
				S11_filtered = S11[path]
				aux_loss = -torch.zeros(1)	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.abs(G11 + (S11.max() - S11.min()))
			print(aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = (G11_filtered.max() - S11_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = (G11_filtered.max() - S11_filtered.max())
	# needs to be fixed
	else:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = (G11_filtered.max() - S11_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = (G11_filtered.max() - S11_filtered.max())
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			#print("Not implemented yet")
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22

	else:
		return aux_loss

def dSz_point_diff(data, z, Ut, bins = 50, bandwidth = 1.0, output_thermo = False, b = 1.0, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0 :
				S11_filtered = S11[path][nan_filter]
				path_sum = S11_filtered.max() - S11_filtered.min()
				aux_loss =  path_sum
			else:
				S11_filtered = S11[path]
				aux_loss = torch.zeros(1)	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = S11.max() - S11.min()
			print(aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			aux_loss = torch.abs((S11_filtered.max() - S11_filtered.min()) - (S22_filtered.max() - S22_filtered.min()))
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			aux_loss = torch.abs((S11_filtered.max() - S11_filtered.min()) - (S22_filtered.max() - S22_filtered.min()))
	# needs to be fixed
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(grad_list[i], dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			print("Not implemented yet")
	else:
		return aux_loss

def dSz_log(data, z, Ut, bins = 50, bandwidth = 1.0, output_thermo = False, b = 1.0, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0 :
				S11_filtered = S11[path][nan_filter]
				path_sum = torch.log(torch.abs(S11_filtered.max() - S11_filtered.min()))
				aux_loss =  path_sum
			else:
				S11_filtered = S11[path]
				aux_loss = torch.log(1e-9 * torch.ones(1))	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = S11.max() - S11.min()
			print(aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			aux_loss = torch.log(torch.abs((S11_filtered.max() - S11_filtered.min()) / ((S11_filtered.max() - S11_filtered.min()) + (S22_filtered.max() - S22_filtered.min()))))
			#print(aux_loss)
		else:
			S22_filtered = 1e-9 * torch.ones(bins)
			aux_loss = torch.log(torch.abs((S11_filtered.max() - S11_filtered.min()) / ((S11_filtered.max() - S11_filtered.min()) + (S22_filtered.max() - S22_filtered.min()))))
	# needs to be fixed
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(grad_list[i], dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			print("Not implemented yet")
	else:
		return aux_loss

def dSz_point_ratio(data, z, Ut, bins = 50, bandwidth = 0.025, output_thermo = False):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, dzdx_list, dzdy_list = jacobian.torch_jacobian_1d(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		#G1 = G1[nan_filter]
		try:
			ref = G1[nan_filter].min()
		except RuntimeError:
			ref = G1.min()

		G11 = G1 - ref

		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		#Uz = Uz[nan_filter]
		try:
			U11 = Uz - Uz[G1 == ref]
		except RuntimeError:
			U11 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S11 = (G11 - b * U11)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,0].min()
		z_std = z[:,0].std()
		z_max = z[:,0].max()


		path = ~((zbins >= float(z_min + z_std)) ^ (zbins <= float(z_max - z_std)))
		path = torch.ravel(path)

		#with open('debug.txt', "w+") as f:
		#	f.write(str(S11) + '\n')
		#	f.write(str(path) + '\n')


		# find the entropy difference, with the endpoint exclueded
		# to help minimize the effect of noise
		path_sum = S11[path].max() - S11[path].min()
		aux_loss =  path_sum

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		try:
			ref = G1[nan_filter].min()
		except RuntimeError:
			ref = G1.min()

		G11 = G1 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		try:
			U11 = Uz - Uz[G1 == ref]
		except RuntimeError:
			U11 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S11 = (G11 - b * U11)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,0].min()
		z_std = z[:,0].std()
		z_max = z[:,0].max()


		path = ~((zbins >= float(z_min + z_std)) ^ (zbins <= float(z_max - z_std)))
		path = torch.ravel(path)

		#with open('debug.txt', "w+") as f:
		#	f.write(str(S11) + '\n')
		#	f.write(str(path) + '\n')


		# find the entropy difference, with the endpoint exclueded
		# to help minimize the effect of noise
		path_sum1 = S11[path].max() - S11[path].min()
				
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[1], dzdy_list[1]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path = ~((zbins >= float(z_min + z_std)) ^ (zbins <= float(z_max - z_std)))
		path = torch.ravel(path)

		#with open('debug.txt', "w+") as f:
		#	f.write(str(S11) + '\n')
		#	f.write(str(path) + '\n')


		# find the entropy difference, with the endpoint exclueded
		# to help minimize the effect of noise
		path_sum2 = S22[path].max() - S22[path].min()
		aux_loss = torch.abs(path_sum1 / (path_sum1 + path_sum2))
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(torch.column_stack([dzdx_list[i], dzdy_list[i]]), dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			print("Not implemented yet")
	else:
		return aux_loss

def dSz_point_both(data, z, Ut, bins = 50, bandwidth = 0.025, output_thermo = False):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, dzdx_list, dzdy_list = jacobian.torch_jacobian_1d(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		#G1 = G1[nan_filter]
		try:
			ref = G1[nan_filter].min()
		except RuntimeError:
			ref = G1.min()

		G11 = G1 - ref

		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		#Uz = Uz[nan_filter]
		try:
			U11 = Uz - Uz[G1 == ref]
		except RuntimeError:
			U11 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S11 = (G11 - b * U11)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,0].min()
		z_std = z[:,0].std()
		z_max = z[:,0].max()


		path = ~((zbins >= float(z_min + z_std)) ^ (zbins <= float(z_max - z_std)))
		path = torch.ravel(path)

		#with open('debug.txt', "w+") as f:
		#	f.write(str(S11) + '\n')
		#	f.write(str(path) + '\n')


		# find the entropy difference, with the endpoint exclueded
		# to help minimize the effect of noise
		path_sum = S11[path].max() - S11[path].min()
		aux_loss =  path_sum

	# right now, the code has not been changed for the cases where the latent space is
	# more than one-dimensional
	elif z.shape[1] == 2:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		try:
			ref = G1[nan_filter].min()
		except RuntimeError:
			ref = G1.min()

		G11 = G1 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		try:
			U11 = Uz - Uz[G1 == ref]
		except RuntimeError:
			U11 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S11 = (G11 - b * U11)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,0].min()
		z_std = z[:,0].std()
		z_max = z[:,0].max()


		path = ~((zbins >= float(z_min + z_std)) ^ (zbins <= float(z_max - z_std)))
		path = torch.ravel(path)

		#with open('debug.txt', "w+") as f:
		#	f.write(str(S11) + '\n')
		#	f.write(str(path) + '\n')


		# find the entropy difference, with the endpoint exclueded
		# to help minimize the effect of noise
		path_sum1 = S11[path].max() - S11[path].min()
				
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[1], dzdy_list[1]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		#G1 = G1[nan_filter]
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref

		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		#Uz = Uz[nan_filter]
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path = ~((zbins >= float(z_min + z_std)) ^ (zbins <= float(z_max - z_std)))
		path = torch.ravel(path)


		# find the entropy difference, with the endpoint exclueded
		# to help minimize the effect of noise
		path_sum2 = b * U22[path].max()
		aux_loss = path_sum1 + path_sum2
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(torch.column_stack([dzdx_list[i], dzdy_list[i]]), dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			print("Not implemented yet")
	else:
		return aux_loss

def dUz(data, z, Ut, bins = 50, bandwidth = 0.01):
	z, dzdx_list, dzdy_list = jacobian.torch_jacobian_1d(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')

	if z.shape[1] == 1:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, random_sample = False)
		G1 = -torch.log(pz)
		ref = G1.argmin()
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		U11 = Uz - Uz[ref]
		aux_loss = U11.max()
		
	elif z.shape[1] == 2:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		#pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, random_sample = False)
		#G1 = -torch.log(pz)
		#ref = G1.argmin()
		#G11 = G1 - G1[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		U11 = Uz - Uz.min() #Uz[ref]
		
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[1], dzdy_list[1]]), dim = 1)
		#pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, random_sample = False)
		#G2 = -torch.log(pz)
		#ref = G2.argmin()
		#G22 = G2 - G2[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		U22 = Uz - Uz.min() #[ref]
		#aux_loss = torch.abs(U11.max() - U11.min() - U22.max() + U22.min())
		aux_loss = torch.abs(U11.max() - U11.min() / (U11.max() - U11.min() + U22.max() - U22.min()))
	else:
		U_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(torch.column_stack([dzdx_list[i], dzdy_list[i]]), dim = 1)
			#pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			#G1 = -torch.log(pz)
			#ref = G1.argmin()
			#G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			U11 = Uz - Uz.min() #Uz[ref]
		aux_loss = torch.abs(U_list[0] / (U_list[1:].sum()))
	return aux_loss

def dUz_path(data, z, Ut, labels, s1 = 0, s2 = 9, bins = 50, bandwidth = 0.01, output_thermo = False):
	# normalize z
	z = (z - z.min()) / (z.max() - z.min())
	z, dzdx_list, dzdy_list = jacobian.torch_jacobian_1d(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		#G1 = G1[nan_filter]
		try:
			ref = G1[nan_filter].min()
		except RuntimeError:
			while True:
				bandwidth += 0.01
				pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
				G1 = -torch.log(pz)
				# filter NaNs
				nan_filter = ~torch.isnan(G1)
				if torch.all(torch.isnan(G1)):
					pass
				else:
					break
			ref = G1[nan_filter].min()
			#ref = G1.min()

		G11 = G1 - ref

		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		#Uz = Uz[nan_filter]
		try:
			U11 = Uz - Uz[G1 == ref]
		except RuntimeError:
			U11 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S11 = (G11 - U11)

		# pick out the path to 'inflate'
		dz = torch.abs((zbins[1] - zbins[0]).ravel())
		labels = torch.argmax(labels, axis = 1)

		# select endpoints plus a buffer of one standard deviation
		z1 = z[:,0][labels == s1].mean()
		z1_std = z[:,0][labels == s1].std()
		z2 = z[:,0][labels == s2].mean()
		z2_std = z[:,0][labels == s2].std()

		if (len(z[:,0][labels == s1]) == 0) or (len(z[:,0][labels == s2]) == 0):
			path = (zbins.ravel() == zbins.ravel())
		else:
			# select the path by excluding the endpoints
			if z1 < z2:
				path = ~((zbins >= float(z1 + z1_std)) ^ (zbins <= float(z2 - z2_std)))
			else:
				path = ~((zbins >= float(z2 + z2_std)) ^ (zbins <= float(z1 - z1_std)))
			path = torch.ravel(path)

		#with open('debug.txt', "w+") as f:
		#	f.write(str(S11) + '\n')
		#	f.write(str(path) + '\n')


		# find the total entropy along the path between endpoints to 'inflate'
		path_sum = (dz * U11[path]).sum()
		aux_loss =  path_sum

	# right now, the code has not been changed for the cases where the latent space is
	# more than one-dimensional
	elif z.shape[1] == 2:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		G1 = G1[nan_filter]
		ref = G1.argmin()
		G11 = G1 - G1[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		Uz = Uz[nan_filter]
		U11 = Uz - Uz[ref]
		S11 = (G11 - U11)
		
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[1], dzdy_list[1]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		G2 = G2[nan_filter]
		ref = G2.argmin()
		G22 = G2 - G2[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		Uz = Uz[nan_filter]
		U22 = Uz - Uz[ref]
		S22 = (G22 - U22)
		aux_loss = torch.abs(S11.max() - S11[ref] - S22.max() + S22[ref])
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(torch.column_stack([dzdx_list[i], dzdy_list[i]]), dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		return aux_loss, G11, U11, S11
	else:
		return aux_loss

def dUz_point(data, z, Ut, bins = 50, bandwidth = 0.025, b = 1.0, output_thermo = False, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0 :
				S11_filtered = S11[path][nan_filter]
				path_sum = torch.abs(G11 + (S11_filtered.max() - S11_filtered.min()))
				aux_loss =  -path_sum
			else:
				S11_filtered = S11[path]
				aux_loss = -torch.zeros(1)	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.abs(G11 + (S11.max() - S11.min()))
			print(aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = (G22_filtered.max() - b * U22_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = (G22_filtered.max() - b * U22_filtered.max())
	# needs to be fixed
	else:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = (G11_filtered.max() - S11_filtered.max()) + (G22_filtered.max() - b * U22_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = (G11_filtered.max() - S11_filtered.max()) + (G22_filtered.max() - b * U22_filtered.max())
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			#print("Not implemented yet")
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22

	else:
		return aux_loss

def dGz(data, z, Ut, bins = 50, bandwidth = 0.1):
	z, dzdx_list, dzdy_list = jacobian.torch_jacobian_1d(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')

	if z.shape[1] == 1:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, random_sample = False)
		G1 = -torch.log(pz)
		ref = G1.argmin()
		#Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G11 = G1 - G1[ref]
		aux_loss = G11.max()
		
	elif z.shape[1] == 2:
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[0], dzdy_list[0]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, random_sample = False)
		G1 = -torch.log(pz)
		ref = G1.argmin()
		G11 = G1 - G1[ref]
		#Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G11 = G1 - G1.min() #Uz[ref]
		
		jac = torch.linalg.norm(torch.column_stack([dzdx_list[1], dzdy_list[1]]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, random_sample = False)
		G2 = -torch.log(pz)
		ref = G2.argmin()
		G22 = G2 - G2[ref]
		#Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		#U22 = Uz - Uz.min() #[ref]
		aux_loss = torch.abs(G11.max() - G22.max())
	else:
		G_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(torch.column_stack([dzdx_list[i], dzdy_list[i]]), dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			G1 = -torch.log(pz)
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			#Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			#U11 = Uz - Uz.min() #Uz[ref]
			G_list.append(G11.max())
		aux_loss = torch.abs(G_list[0] / (G_list[1:].sum()))
	return aux_loss

def dSz_binned(data, z, Ut, bins = 50, bandwidth = 0.1):
	if z.shape[1] == 1:
		z1_hist, dzdx1, dzdy1, dzdx1_traj, dzdy1_traj = jacobian.torch_jacobian_1d_binned(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')
		jac = torch.linalg.norm(torch.column_stack([dzdx1_traj, dzdy1_traj]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, random_sample = False)
		G1 = -torch.log(pz)
		ref = G1.argmin()
		G11 = G1 - G1[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		U11 = Uz - Uz[ref]
		S11 = (G11 - U11)
		aux_loss = S11.max() - S11.min()
		
	elif z.shape[1] == 2:
		z1_hist, dzdx1, dzdy1, dzdx1_traj, dzdy1_traj = jacobian.torch_jacobian_1d_binned(data, z[:,0][:,None], z_bins = bins, nbins = bins, print_figures = False, savepath = './')
		jac = torch.linalg.norm(torch.column_stack([dzdx1_traj, dzdy1_traj]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, random_sample = False)
		G1 = -torch.log(pz)
		ref = G1.argmin()
		G11 = G1 - G1[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		U11 = Uz - Uz[ref]
		S11 = (G11 - U11)
		
		z1_hist, dzdx1, dzdy1, dzdx1_traj, dzdy1_traj = jacobian.torch_jacobian_1d_binned(data, z[:,1][:, None], z_bins = bins, nbins = bins, print_figures = False, savepath = './')
		jac = torch.linalg.norm(torch.column_stack([dzdx1_traj, dzdy1_traj]), dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, random_sample = False)
		G2 = -torch.log(pz)
		ref = G2.argmin()
		G22 = G2 - G2[ref]
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		U22 = Uz - Uz[ref]
		S22 = (G22 - U22)
		aux_loss = torch.abs(S11.max() - S11.min() - S22.max() + S22.min())
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			z1_hist, dzdx1, dzdy1, dzdx1_traj, dzdy1_traj = jacobian.torch_jacobian_1d_binned(data, z1, z_bins = bins, nbins = bins, print_figures = False, savepath = './')
			jac = torch.linalg.norm(torch.column_stack([dzdx1_traj, dzdy1_traj]), dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			G1 = -torch.log(pz)
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
			
	return aux_loss

def dGz_point(data, z, Ut, bins = 50, bandwidth = 1.0, output_thermo = False, b = 1.0, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			aux_loss = G11[path].max()	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = G11.max()
			print(aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			path_sum = G11[path].max()
			#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = torch.zeros(1)
			print(G11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(grad_list[i], dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			print("Not implemented yet")
	else:
		return aux_loss

def dUz_min(data, z, Ut, bins = 50, bandwidth = 0.025, b = 1.0, output_thermo = False, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			# filter NaNs

			nan_filter = ~torch.isnan(U11[path])

			if nan_filter.sum() != 0 :
				U11_filtered = U11[path][nan_filter]
				path_sum = b * (U11_filtered.max() - U11_filtered.min())
				aux_loss =  -path_sum
			else:
				U11_filtered = U11[path]
				aux_loss = torch.zeros(1)	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -b * (U11.max() - U11.min())
			#print(aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				U11_filtered = U11[path][nan_filter]
				path_sum = b * (U11_filtered.max() - U11_filtered.min())
				aux_loss =  path_sum
				#print(aux_loss)
			else:
				U11_filtered = torch.zeros(bins)
				aux_loss = -b * (U11_filtered.max() - U11_filtered.min())
				print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

	else:
		print("Not implemented yet")
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(grad_list[i], dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - b * U11)
			S_list[i] = b * U11
		aux_loss = -torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			print("Not implemented yet")
	else:
		return aux_loss

def variational(data, z, Ut, bins = 50, bandwidth = 1.0, output_thermo = False, b = 1.0, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0 :
				S11_filtered = S11[path][nan_filter]
				path_sum = torch.abs(G11 + (S11_filtered.max() - S11_filtered.min()))
				aux_loss =  -path_sum
			else:
				S11_filtered = S11[path]
				aux_loss = -torch.zeros(1)	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.abs(G11 + (S11.max() - S11.min()))
			print(aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = (G11_filtered.max() - S11_filtered.max()) + (G22_filtered.max() - b * U22_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = (G11_filtered.max() - S11_filtered.max()) + (G22_filtered.max() - b * U22_filtered.max())
	# needs to be fixed
	else:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = (G11_filtered.max() - S11_filtered.max()) + (G22_filtered.max() - b * U22_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = (G11_filtered.max() - S11_filtered.max()) + (G22_filtered.max() - b * U22_filtered.max())
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			#print("Not implemented yet")
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
	else:
		return aux_loss

def abs_variational(data, z, Ut, bins = 50, bandwidth = 1.0, output_thermo = False, b = 1.0, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0 :
				S11_filtered = S11[path][nan_filter]
				path_sum = torch.abs(G11 + (S11_filtered.max() - S11_filtered.min()))
				aux_loss =  -path_sum
			else:
				S11_filtered = S11[path]
				aux_loss = -torch.zeros(1)	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.abs(G11 + (S11.max() - S11.min()))
			print(aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = torch.abs((G11_filtered.max() - S11_filtered.max())) + torch.abs((G22_filtered.max() - b * U22_filtered.max()))
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = torch.abs((G11_filtered.max() - S11_filtered.max())) + torch.abs((G22_filtered.max() - b * U22_filtered.max()))
			
	# needs to be fixed
	else:
		S_list = torch.zeros(z.shape[1])
		for i in range(z.shape[1]):
			z1 = z[:,i][:, None]
			jac = torch.linalg.norm(grad_list[i], dim = 1)
			pz, zbins = jacobian.kde_jacobian(z1, jac, bandwidth = bandwidth, random_sample = False)
			# filter NaNs
			nan_filter = ~torch.isnan(G1)
			G1 = G1[nan_filter]
			ref = G1.argmin()
			G11 = G1 - G1[ref]
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z1, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False)
			Uz = Uz[nan_filter]
			U11 = Uz - Uz[ref]
			S11 = (G11 - U11)
			S_list[i] = S11
		aux_loss = torch.abs(S_list[0] / (S_list[1:].sum()))
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			#print("Not implemented yet")
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
	else:
		return aux_loss

def delta_variational(data, z, Ut, bins = 50, bandwidth = 1.0, output_thermo = False, b = 1.0, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0 :
				S11_filtered = S11[path][nan_filter]
				aux_loss = -torch.abs(S11_filtered.max())
			else:
				S11_filtered = S11[path]
				aux_loss = -torch.zeros(1)	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = torch.abs((G11 - S11)).max()

		#np.savetxt('aux_loss.txt', aux_loss)

	elif z.shape[1] == 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = b * (U11_filtered.max() - U22_filtered.max()) + (S22_filtered.max() - S11_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = b * (U11_filtered.max() - U22_filtered.max()) + (S22_filtered.max() - S11_filtered.max())
	# needs to be fixed
	else:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = b * (U11_filtered.max() - U22_filtered.max()) + (S22_filtered.max() - S11_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = b * (U11_filtered.max() - U22_filtered.max()) + (S22_filtered.max() - S11_filtered.max())
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			#print("Not implemented yet")
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
	else:
		return aux_loss


def logsumexp_variational(data, z, Ut, bins = 50, bandwidth = 1.0, output_thermo = False, b = 1.0, weights = None):
	# load beta from the file 'b.txt'
	# if b.txt does not exist, assume the user does not care
	# about the temperature of the simulation or that the 
	# energy units are arbitrary or kT = 1
	try:
		b = float(np.loadtxt('b.txt'))
	except FileNotFoundError:
		b = 1.0

	z, grad_list = jacobian.torch_jacobian_1d(data, z, all_inputs = True, z_bins = bins, nbins = bins, print_figures = False, savepath = './')


	if z.shape[1] == 1:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		# the free energy is already normalized by 1 / kT
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref

			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			#Uz = Uz[nan_filter]
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			#with open('debug.txt', "w+") as f:
			#	f.write(str(S11) + '\n')
			#	f.write(str(path) + '\n')

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0 :
				S11_filtered = S11[path][nan_filter]
				path_sum = torch.abs(G11 + (S11_filtered.max() - S11_filtered.min()))
				aux_loss =  -path_sum
			else:
				S11_filtered = S11[path]
				aux_loss = -torch.zeros(1)	

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.abs(G11 + (S11.max() - S11.min()))
			print(aux_loss)

	elif z.shape[1] >= 2:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = (0.1 * torch.log(torch.sum(torch.exp(G11_filtered / 0.1))) - 0.1 * torch.log(torch.sum(torch.exp(S11_filtered / 0.1)))) + (0.1 * torch.log(torch.sum(torch.exp(G22_filtered / 0.1))) - b * 0.1 * torch.log(torch.sum(torch.exp(U22_filtered / 0.1))))
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = (0.1 * torch.log(torch.sum(torch.exp(G11_filtered / 0.1))) - 0.1 * torch.log(torch.sum(torch.exp(S11_filtered / 0.1)))) + (0.1 * torch.log(torch.sum(torch.exp(G22_filtered / 0.1))) - b * 0.1 * torch.log(torch.sum(torch.exp(U22_filtered / 0.1))))
	# needs to be fixed
	else:
		jac = torch.linalg.norm(grad_list[0], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,0][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G1 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G1)
		if nan_filter.sum() != 0:
			ref = G1[nan_filter].min()
			G11 = G1 - ref
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,0][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)

			# select endpoints plus a buffer of one standard deviation
			z_min = z[:,0].min()
			z_std = z[:,0].std()
			z_max = z[:,0].max()


			path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
			path = torch.ravel(path)

			# filter NaNs

			nan_filter = ~torch.isnan(S11[path])

			if nan_filter.sum() != 0:
				S11_filtered = S11[path][nan_filter]
				G11_filtered = G11[path][nan_filter]
				U11_filtered = U11[path][nan_filter]
				#print(aux_loss)
			else:
				S11_filtered = torch.zeros(bins)
				G11_filtered = torch.zeros(bins)
				U11_filtered = torch.zeros(bins)
				#aux_loss = S11_filtered.max() - S11_filtered.min()
				#print(aux_loss)

		else:
			G11 = G1 - G1.min()
			ref = G1.argmin()
			Uz, dummy = jacobian.avg_U_KDE_jacobian(z, Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
			try:
				U11 = Uz - Uz[G1 == ref]
			except RuntimeError:
				U11 = Uz - Uz.min()
				#print(Uz, G1, G1 == ref)
			S11 = (G11 - b * U11)
			aux_loss = -torch.zeros(1)
			print(S11, aux_loss)

		# still need to calculate the thermodynamic profile of the 
		# second RC for post-processing even though it does not enter
		# into the calculation of the loss function
		jac = torch.linalg.norm(grad_list[1], dim = 1)
		pz, zbins = jacobian.kde_jacobian(z[:,1][:, None], jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		G2 = -torch.log(pz)
		# filter NaNs
		nan_filter = ~torch.isnan(G2)
		try:
			ref = G2[nan_filter].min()
		except RuntimeError:
			ref = G2.min()

		G22 = G2 - ref
		Uz, dummy = jacobian.avg_U_KDE_jacobian(z[:,1][:, None], Ut, jac, bandwidth = bandwidth, bins = bins, random_sample = False, weights = weights)
		try:
			U22 = Uz - Uz[G2 == ref]
		except RuntimeError:
			U22 = Uz - Uz.min()
			#print(Uz, G1, G1 == ref)
		S22 = (G22 - b * U22)

		# select endpoints plus a buffer of one standard deviation
		z_min = z[:,1].min()
		z_std = z[:,1].std()
		z_max = z[:,1].max()


		path1 = path
		nan_filter1 = nan_filter
		path = ~((zbins >= float(z_min + 0.5 * z_std)) ^ (zbins <= float(z_max - 0.5 * z_std)))
		path = torch.ravel(path)
		# filter NaNs

		nan_filter = ~torch.isnan(S22[path])

		if nan_filter.sum() != 0:
			S22_filtered = S22[path][nan_filter]
			G22_filtered = G22[path][nan_filter]
			U22_filtered = U22[path][nan_filter]
			aux_loss = (G11_filtered.max() - S11_filtered.max()) + (G22_filtered.max() - b * U22_filtered.max())
			#print(aux_loss)
		else:
			S22_filtered = torch.zeros(bins)
			U22_filtered = torch.zeros(bins)
			G22_filtered = torch.zeros(bins)
			aux_loss = (G11_filtered.max() - S11_filtered.max()) + (G22_filtered.max() - b * U22_filtered.max())
	if output_thermo:
		if z.shape[1] == 1:
			return aux_loss, G11, b * U11, S11
		elif z.shape[1] == 2:
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
		else:
			#print("Not implemented yet")
			return aux_loss, G11, b * U11, S11, G22, b * U22, S22
	else:
		return aux_loss
