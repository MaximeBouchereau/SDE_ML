import warnings

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

import torch
import torch.optim as optim
import torch.nn as nn
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point

from itertools import product
import statistics

import sys
import time
import datetime
from datetime import datetime as dtime

# DICTIONNARIES FOR PARAMETERS

params = {'name': "Langevin", 'dimension':1, 'Lambda':1, 'sigma':0.1, 'h_range':[0.001, 0.1], 'Domain_Length':2 , 'Data_Number':100, 'Monte_Carlo_Simulations':1000, 'p_train':0.8, 'Neurons': 100, 'Hidden_Layers': 3}


# PRINT OF PARAMETERS

print(150 * "_")
print(" ")
print(" ### MACHINE LEARNING FOR SDE'S WITH GIBBS'S FINAL MEASURE ###")
print(150 * "_")

print(" ")
print("   PARAMETERS")
print(" ")
print("      > Scientific parameters:")
print("         - Name of the SDE:", params['name'])
print("         - Dimension of the problem:", params['dimension'])
print("         - Lambda:", params['Lambda'])
print("         - sigma:", params['sigma'])
print("         - Step size:", params['h_range'])
print("         - Domain: ", [-params['Domain_Length'], params['Domain_Length']], "^", params['dimension'] )
print(" ")
print("      > ML Parameters:")
print("         - Data Number:", params['Data_Number'])
print("         - Monte-Carlo Simulations:", params['Monte_Carlo_Simulations'])
print("            -> Data [Total]:", params['Data_Number'] * params['Monte_Carlo_Simulations'])
print("         - Proportion of training data:", params['p_train'])
print("         - Neurons per Hidden Layer:", params['Neurons'])
print("         - Hidden Layers:", params['Hidden_Layers'])
print(150 * "_")

# CODE BEGINS HERE


class VectorField:
    """Class for Vector Fields."""

    def V(self, x):
        """Potential function.

        Inputs:
        - x: Tensor of shape (d,K) where d is the dimension of the space associated to the solution
        of the SDE and K is the number of evaluations."""

        VV = x[0, :] ** 2
        VV = VV.unsqueeze(0)

        return VV

    def f(self, x):
        """Vector field corresponding to the gradient of the potential.

        Inputs:
        - x: Tensor of shape (d,N) where d is the dimension of the space associated to the solution
        of the SDE and N is the number of evaluations."""

        x.requires_grad = True
        VV = self.V(x)
        nabla_V = torch.autograd.grad(outputs=VV, inputs=x, grad_outputs=torch.ones_like(VV), create_graph=True)[0].detach()
        x.requires_grad = False

        return nabla_V

    def sigma(self, x):
        """Vector field corresponding to drift term (additive Gaussian white noise).

        Inputs:
        - x: Tensor of shape (d,N) where d is the dimension of the space associated to the solution
        of the SDE and N is the number of evaluations."""

        Sigma = torch.ones_like(x)

        return Sigma

class Data(VectorField):
    """Class for data creation."""

    def data(self, T, delta_t, N_iter_2):
        """Generates data in order to see Gibb's final measure. Euler-Maruyama [Forward Euler] method for SDE's is used
        to approximate exact solutions.

        Then, generates a second part of the dataset by applying "exact" flow [approximated with Euler-Maruyama method]
        to initial conditions distributed with approximated Gibb's final measure.

        Inputs:
        - T: Float - Final time to reach Gibb's final measure.
        - delta_t: Float - Time step to solve SDE's.
        - N_iter_2: Int - Number of iterations for the generation of the second part of dataset."""

        print(150 * "_")
        print(" ")
        print("   Data creation")
        print(150 * "_")

        # Parameters extraction
        d, L = params['dimension'], params['Domain_Length']
        Lambda, sigma, h_data = params['Lambda'], params['sigma'], params['h_range']
        K, M = params['Data_Number'], params['Monte_Carlo_Simulations']
        p_train = params['p_train']

        # Time interval construction
        TT = torch.arange(0, T, delta_t)

        # Initialization [Space data]
        X = torch.FloatTensor(d, K).uniform_(-L, L)
        X = torch.kron(X, torch.ones(1, M))

        # Initialization [Time steps]
        hh = torch.FloatTensor(d, K).uniform_(torch.log(torch.tensor([h_data[0]]))[0], torch.log(torch.tensor([h_data[1]]))[0])
        hh = torch.exp(hh)
        hh = torch.kron(hh, torch.ones(1, M))

        # SDE resolution [approximation of exact flow] to reach Gibb's final measure
        print(" ")
        for n in range(TT.numel()):
            print("   > Reach Gibb's final measure - Iteration:", n + 1, "/", TT.numel(), end="\r")
            B = delta_t ** 0.5 * torch.randn(d, K * M)
            X = X - delta_t * Lambda * self.f(X) + B * sigma * self.sigma(X)

        # First part of dataset: Initial conditions distributed with Gibb's final measure
        X0 = X[:]

        # Generation of Second part of dataset: Exact flow with initial conditions belonging to first part pf dataset.
        print(" ")
        print(" ")
        for n in range(N_iter_2):
            print("   > Compute exact flow to first part of dataset - Iteration:", n + 1, "/", N_iter_2, end="\r")
            B = (hh / N_iter_2) ** 0.5 * torch.randn(d, K * M)
            X = X - (hh / N_iter_2) * Lambda * self.f(X) + B * sigma * self.sigma(X)

        # Second part of dataset
        X1 = X[:]

        # Split between train and test datasets
        print(" ")
        print(" ")
        print("   > Split between train and test datasets")
        K0 = int(p_train * K)

        X0_train, X1_train, h_train, B_train = X0[:, 0: K0*M], X1[:, 0: K0*M], hh[:, 0: K0*M], B[:, 0: K0*M]
        X0_test, X1_test, h_test, B_test = X0[:, K0*M: ], X1[:, K0*M: ], hh[:, K0*M: ], B[:, K0*M: ]

        print(" ")
        print("   > Save dataset")
        path = r"C:\Documents\Implementaitons\Other_projects\Datasets"
        name_dataset = "\DATA_SDE_" + str(params['name']) + "_dim=" + str(params['dimension']) + "_Len=" + str(params['Domain_Length']) + "_Lambda=" + str(params['Lambda']) + "_sigma=" + str(params['sigma']) + "_h=" + str(params['h_range']) + "_NData=" + str(params['Data_Number']) + "_NMC=" + str(params['Monte_Carlo_Simulations']) + ".npy"
        torch.save((X0_train, X0_test, X1_train, X1_test, h_train, h_test, B_train, B_test, params), path + name_dataset)
        print(150 * "_")

        return None

class NN(nn.Module, Data):
    """Neural Network architecture used for our problem"""

    def __init__(self):
        super().__init__()
        d, zeta, HL = params['dimension'], params['Neurons'], params['Hidden_Layers']
        self.R_f_modif = nn.ModuleList([nn.Linear(d + 1, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])
        self.R_sigma_modif = nn.ModuleList([nn.Linear(d + 1, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])

    def forward(self, x, h):
        """Structured Neural Network.
        Inputs:
         - x: Tensor of shape (d,n) - space variable
         - h: Tensor of shape (1,n) - Step size"""

        x = x.float().T
        h = h.float().T

        # Structure of the solution of modified fields involved in the SDE

        x_R_f = torch.cat((x, h), dim=1)
        x_R_sigma = torch.cat((x, h), dim=1)

        for i, module in enumerate(self.R_f_modif):
            x_R_f = module(x_R_f)

        for i, module in enumerate(self.R_sigma_modif):
            x_R_sigma = module(x_R_sigma)

        x_f = self.f(x.detach().T).T + h * x_R_f
        x_sigma = self.sigma(x.detach().T).T + h * x_R_sigma

        return (x_f).T, (x_sigma).T

class Train(NN):
    """Class for Neural Network training"""

    def Loss(self, params_data, X0, X1, h, B, model):
        """Computes the Loss function between two series of data X0 and X1. d is the dimension of the problem

        Inputs:
        - params_data: Dict - Dictionary containing parameters of data
        - X0: Tensor of shape (d, b) - Inputs of the Neural Network.
        - X1: Tensor of shape (d, b) - Outputs of the Neural Network.
        - h: Tensor of shape (1, b) - Step sizes associated to SDE.
        - B: Tensor of shape (d, b) - Brownian motion [Wiener process]
        - model: NN which will be optimized

        where:
        > b is the size of the batch.
        > d is the dimension of the problem: params_data['dimension']

        => Computes a predicted value X1_hat which is a tensor of shape (b, d) and returns the error between X1_hat and X1 and returns a tensor of shape (1,1)"""

        d, M = params_data['dimension'], params_data['Monte_Carlo_Simulations']
        Lambda, sigma = params_data['Lambda'], params_data['sigma']

        # Number of data [without Monte-Carlo]
        K = h.numel() // M

        X0, X1, h, B = torch.tensor(X0, dtype=torch.float32), torch.tensor(X1, dtype=torch.float32), torch.tensor(h, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)
        X0.requires_grad, X1.requires_grad = True, True

        f_theta, sigma_theta = model(X0, h)
        X1_hat = X0 - h * Lambda * f_theta + B * sigma * sigma_theta

        # Data transform to get average over stochastic terms
        X1_hat, X1 = torch.cos(X1_hat), torch.cos(X1)

        X1_hat = torch.transpose((X1_hat.T).reshape(K, M, d), 1, 2)
        X1_hat = X1_hat.mean(dim = 2).T

        X1 = torch.transpose((X1.T).reshape(K, M, d), 1, 2)
        X1 = X1.mean(dim=2).T

        h = torch.transpose((h.T).reshape(K, M, 1), 1, 2)
        h = h.mean(dim=2).T

        loss = (((X1_hat - X1)).abs() ** 1).mean()

        return loss

    def train(self, BS, alpha, N_epochs, N_epochs_print, name_dataset):
        """Makes the training on the dataset

        Inputs:

        - BS: Integer - size of the batches for mini-batching.
        - alpha: Two possibilities:
                > Float - Learning rate, same for all training
                > List [alpha_0, alpha_1] - alpha_0 at the beginning of training and alpha_1 at the end of training. Exponential decay w.r.t. epochs.
        - N_epochs: Integer - Number of epochs for training.
        - N_epochs_print: Integer - Number of epochs between two prints of the value of the Loss.
        - name_dataset: Str - Name of the loaded dataset.

        => Returns the lists Loss_train and Loss_test of the values of the Loss respectively for training and test,
        and best_model, which is the best approximation of the searched mapping for modified fields for SDE.."""

        print(" ")
        print(150 * "_")
        print(" ")
        print("  Training...")
        print(150 * "_")
        print(" ")

        # Data extraction
        X0_train, X0_test, X1_train, X1_test, h_train, h_test, B_train, B_test, params_data = torch.load(name_dataset)

        # Batch size maximum
        BS = min(BS, int(params_data['Data_Number'] * params_data['p_train']))

        # Neural Network training
        model = NN()

        start_time_train = time.time()

        # Optimizer - AdamW Algorithm
        if type(alpha) == float:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=alpha, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=True) # AdamW Algorithm
        elif type(alpha) == list:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=alpha[0], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=True) # AdamW Algorithm

        # Initialization of model, loss_train and loss_test
        best_model, best_loss_train, best_loss_test = model, torch.inf, torch.inf  # Gets the best model (which is the best minimizer of the loss function)
        Loss_train = []  # List for loss_train values
        Loss_test = []  # List for loss_test values
        optimizer.zero_grad()

        # Training process
        for epoch in range(N_epochs + 1):
            # optimizer.zero_grad()
            # for ixs in torch.split(torch.arange(x_train.shape[0]), BS):
            # for ixs in torch.randperm(x_train.shape[0])[:BS]:
            ixs = torch.randperm(int(params_data['Data_Number'] * params_data['p_train']))[:BS]
            ixs = torch.kron(params_data['Monte_Carlo_Simulations'] * ixs, torch.ones(params_data['Monte_Carlo_Simulations']))
            ixs = torch.kron(torch.ones(BS), torch.arange(0, params_data['Monte_Carlo_Simulations'], 1)) + ixs
            ixs = torch.tensor(ixs, dtype=int)
            model.train()
            X0_batch = X0_train[:, ixs]
            X1_batch = X1_train[:, ixs]
            h_batch = h_train[:, ixs]
            B_batch = B_train[:, ixs]
            loss_train = self.Loss(params_data, X0_batch, X1_batch, h_batch, B_batch, model)
            loss_train.backward()
            optimizer.step()  # Optimizer goes to the next epoch for gradient descent

            if type(alpha) == list:
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=alpha[0] * (alpha[1] / alpha[0]) ** (epoch / N_epochs), betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=True) # AdamW Algorithm - update

            loss_train = self.Loss(params_data, X0_train, X1_train, h_train, B_train, model)
            loss_test = self.Loss(params_data, X0_test, X1_test, h_test, B_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            # Print of the loss values and estimation of remaining time
            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')

                print('    Step', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)

        print(" ")

        print("Loss_train [final]=", format(best_loss_train, '.4E'))
        print("Loss_test [final]=", format(best_loss_test, '.4E'))

        print("Computational time for training [h:min:s]:", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        # Saving of the parameters of the training of our model
        model_params = params_data.copy()
        model_params['Batch_Size'] = BS
        model_params['N_epochs'] = N_epochs
        model_params['Learning_Rate[s]'] = alpha

        path = r"C:\Documents\Implementaitons\Other_projects\Models"
        name_model = r"\Learned_Modified_Field_SDE"
        torch.save( (Loss_train, Loss_test, best_model, model_params), path + name_model)

        print(150 * "_")

        return None

