# utility functions required for running the Jacobian code. 

import numpy as np
from scipy import optimize
import torch
def list_enumerate(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, *elem
        n += 1

def make_fes2d(x, y, bins = 50):
    hist, xbins, ybins = np.histogram2d(x, y, bins = bins)
    dx = xbins[1] - xbins[0]
    dy = ybins[1] - ybins[0]
    xbins = (xbins[1:] + xbins[:-1]) / 2
    ybins = (ybins[1:] + ybins[:-1]) / 2
    hist = hist / (hist.sum()*dx*dy)
    fes = -np.log(hist.T)
    return fes

def fuzzy_histogram(data, f, bin_centers, binwidth = 1):
    hist = np.zeros(len(bin_centers))
    counter = np.zeros_like(hist)
    for i, bin_center in enumerate(bin_centers):
        #print(i, bin_center)
        for k, x in enumerate(data):
            hist[i] += f[k] * np.exp(-(x - bin_center)**2/ (2 * binwidth*binwidth))
            counter[i] += np.exp(-(x - bin_center)**2/ (2 * binwidth*binwidth))
    return hist, counter

def fit_sigmoid(x, y):
    popt, pcov = optimize.curve_fit(sigmoid, x, y)
    a, b, c, d, e, f = popt; print(*popt)
    return sigmoid(x, a, b, c, d, e, f), popt

def sigmoid(x, a, b, c, d, e, f): 
    return a + (b - a) / ((c + d * np.exp(-e * x))**f)

def dsigmoid(x, a, b, c, d, e, f):
    return (b - a) * (f * e * d * np.exp(-e * x) * (1 + d * np.exp(-e * x))**(-f - 1))

def energetic_double_well_potential(x, y):
    return (x**2 - 1)**2 + y**2

def entropic_double_well_potential(x, y, sigma_x = 0.1, sigma_y = 0.1):
    return x**6 + y**6 + torch.exp(-(y /sigma_y)**2) * (1 - torch.exp(-(x / sigma_x)**2))

def a(x, delta = 0.05, x0 = 0.0):
    return 0.2*(1 + 5*torch.exp(-(x - x0)**2 / delta))**2

def temperature_switch_potential(x, y, hx = 0.5, hy = 1.0, delta = 0.05, x0 = 0.0):
    return hx * (x**2 - 1)**2 + (hy + a(x, delta = delta, x0 = x0)) * (y**2 - 1)**2


def restore_SPIB_model(RC_dim, nn1, nn2, data_shape, d, dt, beta, gamma, lr, seed, 
                       pseudo_dim = 10, output_dim = 10, encoder_type = "Nonlinear", device = "cpu", weighted = False, path = './'):
    import SPIB
    
    if weighted:
        weighted = 'Weighted'
    else:
        weighted = 'Unweighted'
    RC_dim = RC_dim
    pseudo_dim = pseudo_dim
    output_dim = output_dim
    data_shape = (data_shape,)

    neuron_num1 = nn1
    neuron_num2 = nn2
    encoder_type = encoder_type

    device = torch.device(device)

    # Update Label
    ## Remember to update both directories
    UpdateLabel = True
    prefix_00 = path
    representative_inputs_path = prefix_00 + weighted + '_d=%d_t=%d_b=%.4f_gamma=%.4f_learn=%f_representative_inputs%d.npy' % (RC_dim, dt, beta, gamma, lr, seed)
    prefix = prefix_00 + weighted + "_d=%d_t=%d_b=%.4f_gamma=%.4f_learn=%fcpt%d/IB_final_cpt.pt" % (RC_dim, dt, beta, gamma, lr, seed)


    restore_path=prefix
    representative_inputs = torch.tensor(np.load(representative_inputs_path))
    index = 1
    encoder_type = encoder_type
    IB = SPIB.SPIB(encoder_type, RC_dim, output_dim, data_shape, device, UpdateLabel, neuron_num1, neuron_num2)
    IB.reset_representative(representative_inputs)
    checkpoint=torch.load(restore_path)
    IB.load_state_dict(checkpoint['state_dict'])
    return IB
