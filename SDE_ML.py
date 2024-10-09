import warnings

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

import torch
import torch.optim as optim
import torch.nn as nn
import copy

import numpy as np
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

# Python code for SDE's & Machine Learning (adapted from backward error analysis)

dyn_syst = "Black_Scholes"  # Dynamical system studied (choice between "Black_Scholes" and "Pendulum")
num_meth = "Forward Euler"  # Choice of the numerical method ("Forward Euler" and "MidPoint")
Mu = 1.0                    # Parameter for Black & Scholes model
Sigma = 1e-1                # Noise for SDE
step_h = [0.01, 0.5]         # Interval where steps of time are selected for training
T_simul = 1                # Time for ODE's simulation
h_simul = 0.01              # Time step used for ODE's simulation
N_iter = 5                 # Number of iterations for Fiexed-point method

# AI parameters [adjust]

K_data = 100          # Quantity of data
N_data = 100        # Quantity of required data for Law of Large Numbers convergence (approximation of integrals)
R = 2                  # Amplitude of data in space (i.e. space data will be selected in the box [-R,R]^d)
p_train = 0.8          # Proportion of data for training
HL = 2                 # Hidden layers per MLP
zeta = 200             # Neurons per hidden layer
alpha = 1e-3           # Learning rate for gradient descent
Lambda = 1e-9          # Weight decay
BS = K_data * N_data   # Batch size (for mini-batching)
N_epochs = 200         # Epochs
N_epochs_print = 20    # Epochs between two prints of the Loss value

print(150 * "-")
print("Using of MLP's with classes for learning of dynamical system - Modified equation - Variable steps of time")
print(150 * "-")

print("   ")
print(150 * "-")
print("Parameters:")
print(150 * "-")
print('    # Maths parameters:')
print('        - Dynamical system studied:', dyn_syst)
print('        - Numerical method:', num_meth)
print("        - Parameter for Black & Scholes model:", Mu)
print("        - Noise for SDE:", Sigma)
print("        - Interval where steps of time are selected for training:", step_h)
print("        - Time for ODE's simulation:", T_simul)
print("        - Step size used for ODE's simulation:", h_simul)
print("    # AI parameters:")
print("        - Data's number:", K_data)
print("        - Data's number for LLN approximation:", N_data)
print("        - Total for Data:", K_data * N_data)
print("        - Amplitude of data in space:", R)
print("        - Proportion of data for training:", format(p_train, '.1%'))
print("        - Hidden layers per MLP:", HL)
print("        - Neurons on each hidden layer:", zeta)
print("        - Learning rate:", format(alpha, '.2e'))
print("        - Weight decay:", format(Lambda, '.2e'))
print("        - Batch size (mini-batching for training):", BS)
print("        - Epochs:", N_epochs)
print("        - Epochs between two prints of the Loss value:", N_epochs_print)


if dyn_syst == "Black_Scholes":
    d = 1
elif dyn_syst == "Pendulum":
    d = 2
else:
    d = 3

def X_0_start(dyn_syst):
    """Gives the initial data (vector) for SDE's integration and study of trajectories"""
    if d == 2:
        if dyn_syst == "Pendulum":
            return np.array([1.5,0.0])
        else:
            return np.array([0.5,0.5])
    if d == 1:
        return np.array([1.0])

class Vector_field:
    """Vector fields involved"""

    def f_array(self, t, y):
        """Returns the vector field associated to the corresponding dynamical system
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,1) - Space variable"""

        y = np.array(y).reshape(d,)
        #z = np.zeros_like(y)

        # if dyn_syst == "Rigid Body":
        #     z = np.array([(1 / I3 - 1 / I2) * y[1, 0] * y[2, 0], (1 / I1 - 1 / I3) * y[0, 0] * y[2, 0],
        #                   (1 / I2 - 1 / I1) * y[0, 0] * y[1, 0]])
        #
        # if dyn_syst == "SIR":
        #     z = np.array([-(R0 / Tr) * y[0, 0] * y[1, 0], (R0 / Tr) * y[0, 0] * y[1, 0] - (1 / Tr) * y[1, 0],
        #                   (1 / Tr) * y[1, 0]])
        #
        # if dyn_syst == "Stable":
        #     z = np.array([y[1, 0], -y[0, 0], -y[2, 0]])
        #
        # if dyn_syst == "Lotka-Volterra":
        #     z = np.array([y[0, 0] * (beta11 - beta12 * y[1, 0]), y[1, 0] * (beta21 * y[0, 0] - beta22)])
        if dyn_syst == "Black_Scholes":
            z = Mu * y

        if dyn_syst == "Pendulum":
            z = np.array([-np.sin(y[1]), y[0]])

        return z

    def f(self, t, y):
        """Returns the vector field associated to the corresponding dynamical system
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,1) - Space variable"""

        nb_coeff = 1
        for s in y.shape:
            nb_coeff = nb_coeff * s
        y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        z = torch.zeros_like(y)

        # if dyn_syst == "Rigid Body":
        #     z[0, :] = (1 / I3 - 1 / I2) * y[1, :] * y[2, :]
        #     z[1, :] = (1 / I1 - 1 / I3) * y[0, :] * y[2, :]
        #     z[2, :] = (1 / I2 - 1 / I1) * y[0, :] * y[1, :]
        #
        # if dyn_syst == "SIR":
        #     z[0, :] = -(R0 / Tr) * y[0, :] * y[1, :]
        #     z[1, :] = (R0 / Tr) * y[0, :] * y[1, :] - (1 / Tr) * y[1, :]
        #     z[2, :] = (1 / Tr) * y[1, :]
        #
        # if dyn_syst == "Stable":
        #     z[0, :] = y[1, :]
        #     z[1, :] = -y[0, :]
        #     z[2, :] = -y[2, :]
        #
        # if dyn_syst == "Lotka-Volterra":
        #     z[0, :] = y[0, :] * (beta11 - beta12 * y[1, :])
        #     z[1, :] = y[1, :] * (beta21 * y[0, :] - beta22)

        if dyn_syst == "Black_Scholes":
            z = Mu * y

        if dyn_syst == "Pendulum":
            z[0, :] = -np.sin(y[1, :])
            z[1, :] = y[0, :]

        return z

    def f_modif_array(self, t, y, h):
        """Returns the modified vector field at order 3 associated to the corresponding dynamical system
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,1) - Space variable
        - h: Float - Step size"""

        y = np.array(y).reshape(d, )
        if dyn_syst == "Black_Scholes":
            z = (Mu + h * Mu ** 2 / 2 + h ** 2 * Mu ** 3 / 6) * y
        return z

    def sigma_array(self, t, y):
        """Returns the drift field associated to the corresponding dynamical system
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,1) - Space variable"""

        y = np.array(y).reshape(d, )

        if dyn_syst == "Black_Scholes":
            return Sigma * y

        if dyn_syst == "Pendulum":
            return Sigma * self.f_array(t, y)

    def sigma(self, t, y):
        """Returns the drift field associated to the corresponding dynamical system
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,1) - Space variable"""

        nb_coeff = 1
        for s in y.shape:
            nb_coeff = nb_coeff * s
        y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        z = torch.zeros_like(y)

        if dyn_syst == "Black_Scholes":
            z = Sigma * y
            return z

        if dyn_syst == "Pendulum":
            z = Sigma * self.f(t, y)
            return z

    def sigma_modif_array(self, t, y, h):
        """Returns the modified drift field at order 3 associated to the corresponding dynamical system
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,1) - Space variable
        - h: Float - Step size"""

        y = np.array(y).reshape(d, )

        if dyn_syst == "Black_Scholes":
            #return  (Sigma + 1 * h * np.sqrt(Sigma ** 4 / 2 + 2 * Mu * Sigma ** 2) ** 1 + 0 * h ** 2 * np.sqrt(Sigma ** 6 / 6 + Mu * Sigma ** 4 + 2 * Mu ** 2 * Sigma ** 2)) * y
            #return  (Sigma + h * (Sigma ** 3 / 4 + Mu)) * y
            return np.sqrt(Sigma ** 2 + h * (Sigma ** 3 / 2 + 2 * Mu * Sigma ** 2)) * y

class Data(Vector_field):
    """Class for Data creation"""

    def Exact_Flow(self, X_0, t_0, T, h):
        """Computation of exact flow of SDE with Euler Maruyama method with small step size.
        Inputs:
        - X_0: Array of shape (d,) - Initial datum
        - t_0: Float - Starting time
        - T: Float - Length of integration interval.
        - h: Float - Step size"""

        n_it = 20
        delta_t = h/n_it
        TT = np.arange(t_0, t_0 + T + h, h)
        B = np.zeros_like(TT) # Associated brownian trajectory
        X, Y = np.zeros((d,TT.size)), np.zeros((d,TT.size))
        X[:,0], Y[:,0] = X_0, X_0
        B[0] = 0.0
        for n in range(TT.size-1):
            XX, YY = X[:, n], Y[:, n]
            B[n+1] = B[n]
            if dyn_syst == "Black_Scholes":
                BB = np.random.normal(0, h ** 0.5)
                X[:, n + 1] = np.exp(h * (Mu - Sigma ** 2 / 2) + Sigma * BB) * X[:, n]
                Y[:, n + 1] = np.exp(h * (Mu - Sigma ** 2 / 2) - Sigma * BB) * Y[:, n]
                B[n+1] = B[n+1] + BB
            else:
                for j in range(n_it):
                    BB = np.random.normal(0, delta_t**0.5)
                    B[n+1] = B[n+1] + BB
                    if num_meth == "Forward Euler":
                        XX = XX + delta_t*Vector_field().f_array(0.0, XX) + Vector_field().sigma_array(0.0, XX)*BB
                        YY = YY + delta_t*Vector_field().f_array(0.0, YY) - Vector_field().sigma_array(0.0, YY)*BB
                    if num_meth == "MidPoint":
                        def F_iter(y):
                            """Iteration function for MidPoint method via fixed-point method.
                            Input:
                            - y: Array of shape (d,) - Space variable"""
                            return XX + delta_t * (Vector_field().f_array(0.0, y) + Vector_field().f_array(0.0, XX)) / 2 + (Vector_field().sigma_array(0.0, XX) + Vector_field().sigma_array(0.0, y)) * BB / 2
                        def F_iter_bis(y):
                            """Iteration function for MidPoint method via fixed-point method.
                            Input:
                            - y: Array of shape (d,) - Space variable"""
                            return XX + delta_t * (Vector_field().f_array(0.0, y) + Vector_field().f_array(0.0, XX)) / 2 - (Vector_field().sigma_array(0.0, XX) + Vector_field().sigma_array(0.0, y)) * BB / 2
                        xx, yy = XX, YY
                        for k in range(N_iter):
                            xx, yy = F_iter(xx), F_iter_bis(yy)
                        XX, YY = xx, yy

                X[:, n + 1], Y[:, n + 1] = XX, YY

        return X, Y, B

    def Exact_Flow_ODE(self, X_0, t_0, T, h):
        """Computation of exact flow of ODE adapted from SDE (no drift term) with Euler method with small step size.
        Inputs:
        - X_0: Array of shape (d,) - Initial datum
        - t_0: Float - Starting time
        - T: Float - Length of integration interval.
        - h: Float - Step size"""

        n_it = 20
        delta_t = h/n_it
        TT = np.arange(t_0, t_0 + T + h, h)
        B = np.zeros_like(TT) # Associated brownian trajectory
        X, Y = np.zeros((d,TT.size)), np.zeros((d,TT.size))
        X[:,0], Y[:,0] = X_0, X_0
        B[0] = 0.0
        for n in range(TT.size-1):
            XX, YY = X[:, n], Y[:, n]
            B[n+1] = B[n]
            for j in range(n_it):
                BB = np.random.normal(0, delta_t**0.5)
                B[n+1] = B[n+1] + BB
                XX = XX + delta_t*Vector_field().f_array(0.0, XX)
                YY = YY + delta_t*Vector_field().f_array(0.0, YY)
            X[:,n+1], Y[:,n+1] = XX, YY

        return X, Y, B

    def Numerical_Flow(self, X_0, t_0, T, h):
        """Computation of numerical flow flow of SDE with specified numerical method method with small step size.
        Inputs:
        - X_0: Array of shape (d,) - Initial datum
        - t_0: Float - Starting time
        - T: Float - Length of integration interval.
        - h: Float - Step size"""

        TT = np.arange(t_0, t_0 + T + h, h)
        X = np.zeros((d, TT.size))
        X[:, 0] = X_0
        for n in range(TT.size-1):
            XX = X[:, n]
            BB = np.random.normal(0, h**0.5)
            if num_meth == "Forward Euler":
                X[:, n+1] = XX + h*Vector_field().f_array(0.0, XX) + Vector_field().sigma_array(0.0, XX)*BB
            if num_meth == "MidPoint":
                def F_iter(y):
                    """Iteration function for MidPoint method via fixed-point method.
                    Input:
                    - y: Array of shape (d,) - Space variable"""
                    return X[:, n] + h*(Vector_field().f_array(0.0, y) + Vector_field().f_array(0.0, X[:, n]))/2 + (Vector_field().sigma_array(0.0, X[:, n]) + Vector_field().sigma_array(0.0, y))*BB/2
                yy = X[:, n]
                for k in range(N_iter):
                    yy = F_iter(yy)
                X[:, n + 1] = yy
        return X

    def Data_create(self, K=K_data, p=p_train, h_data=step_h):
        """production of a set of initial data X0 and final data X1 = flow-h(X0) associated to the dynamical system y'=f(t,y)
        Inputs:
        - K: Data's number (default: K_data)
        - p: Proportion of data for training (default: p_train)
        - h_data: List - Interval where steps of time are chosen for training (default: step_h)
        Denote K0 := int(p*K) the number of data for training
        => Returns the tuple (X0_train, X0_test, X1_train, X1_test, h_train, h_test, B_train, B_test) where:
            - X0_train is a tensor of shape (d,K0) associated to the initial data for training
            - X0_test is a tensor of shape (d,K-K0) associated to the initial data for test
            - X1_train is a tensor of shape (d,K0) associated to the final data for training
            - X1_test is a tensor of shape (d,K-K0) associated to the final data for test
            - h_train is a tensor of shape (1,K0) associated to data of steps of time for training
            - h_train is a tensor of shape (1,K-K0) associated to data of steps of time for test
            - B_train is a tensor of shape (1,K0) associated to data of brownian trajectories for training
            - B_test is a tensor of shape (1,K0) associated to data of brownian trajectories for test
            Each column of the tensor X1_* correspopnds to the flow at h_* of the same column of X0_*
            Initial data are uniformlyrandomly selected in [-R,R]^d with uniform law (excepted for Rigid Body, in a spherical crown)"""

        start_time_data = time.time()

        print(" ")
        print(150 * "-")
        print("Data creation...")
        print(150 * "-")

        K0 = int(p * K * N_data)
        XX0 = np.random.uniform(low=-R, high=R, size=(d, K_data))
        if dyn_syst == "Black_Scholes":
            XX0 = np.abs(XX0)
        XX0 = np.kron(XX0, np.ones((1,N_data)))
        # if num_meth == "RK2":
        #    beta = 0.4
        #    N = np.linalg.norm(YY0, ord=np.infty, axis=0)
        #    P = np.random.uniform(0, 1, size=(1, K))
        #    YY0 = R * YY0 / N * P ** beta
        XX1 = np.zeros((d, K * N_data))
        BB = np.zeros((1, K * N_data))

        hh = np.exp(np.random.uniform(low=np.log(h_data[0]), high=np.log(h_data[1]), size=(1, K)))
        hh = np.kron(hh, np.ones((1,N_data)))

        if dyn_syst == "Black_Scholes":
            print("Direct computation [explicit solution known]")
            BB = np.random.normal(loc=0, scale=hh**0.5)
            XX1 = XX0 * np.exp((Mu - 0.5 * Sigma ** 2) * hh) * np.exp(Sigma * BB)

        else:
            pow = max([int(np.log10(K*N_data) - 1), 3])
            pow = min([pow, 6])

            for k in range(K):
                for n in range(N_data // 1):
                    k_it = k * N_data + n
                    #k_it = k * N_data + 2 * n
                    end_time_data = start_time_data + (K / (k + 1)) * (time.time() - start_time_data)
                    end_time_data = datetime.datetime.fromtimestamp(int(end_time_data)).strftime(' %Y-%m-%d %H:%M:%S')
                    print(" Loading :  {} % \r".format(str(int(10 ** (pow) * (k_it + 1) / (K*N_data)) / 10 ** (pow - 2)).rjust(3)), " Estimated time for ending : " + end_time_data, " - ", end="")
                    phi = self.Exact_Flow(X_0 = XX0[:, k_it], t_0 = 0, T = 2 * hh[0, k_it], h=hh[0, k_it])
                    XX1[:, k_it] = phi[0][:, 1]
                    #XX1[:, k_it + 1] = phi[1][:, 1]
                    BB[:, k_it] = phi[2][1]
                    #BB[:, k_it + 1] = - phi[2][1]

        X0_train = torch.tensor(XX0[:, 0:K0])
        X0_test = torch.tensor(XX0[:, K0:K*N_data])
        X1_train = torch.tensor(XX1[:, 0:K0])
        X1_test = torch.tensor(XX1[:, K0:K*N_data])
        h_train = torch.tensor(hh[:, 0:K0])
        h_test = torch.tensor(hh[:, K0:K*N_data])
        B_train = torch.tensor(BB[:, 0:K0])
        B_test = torch.tensor(BB[:, K0:K*N_data])

        print("Computation time for data creation (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_data))))
        return (X0_train, X0_test, X1_train, X1_test, h_train, h_test, B_train, B_test)

class NN(nn.Module, Data):
    """Class for Neural Networks"""
    def __init__(self):
        super().__init__()
        self.R_f_modif = nn.ModuleList([nn.Linear(d + 1, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])
        self.R_sigma_modif = nn.ModuleList([nn.Linear(d + 1, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])

    def forward(self, x, h):
        """Structured Neural Network.
        Inputs:
         - x: Tensor of shape (d,n) - space variable
         - h: Tensor of shape (1,n) - Step size"""

        x = x.T
        x = x.float()
        h = torch.tensor(h).T

        # Structure of the solution of modified fields involved in the SDE

        x_R_f = torch.cat((x , h) , dim = 1)
        x_R_sigma = torch.cat((x , h) , dim = 1)
        #x_R = torch.cat((x , h) , dim = 1)

        for i, module in enumerate(self.R_f_modif):
            x_R_f = module(x_R_f)

        for i, module in enumerate(self.R_sigma_modif):
            x_R_sigma = module(x_R_sigma)

        x_f = self.f(0 * h, x.T).T + h * x_R_f
        x_sigma = self.sigma(0 * h, x.T).T + h * x_R_sigma

        return (x_f).T , (x_sigma).T

class Train(NN, Data):
    """Training of the neural network, depends on the selected numerical method
    Choice of the numerical method:
    - Forward Euler
    - MidPoint"""

    def Loss(self, X0, X1, H, B, model):
        """Computes the Loss function between two series of data X0 and X1 according to the numerical method
        Inputs:
        - X0: Tensor of shape (d,n): Initial data
        - X1: Tensor of shape (d,n): Exact data
        - H: Tensor of shape (1,n): Step size
        - B: Tensor of shape (1,n): Brownian trajectory
        - model: Neural network which will be optimized
        Computes a predicted value Y1hat which is a tensor of shape (d,n) and returns the mean squared error between Y1hat and Y1
        => Returns a tensor of shape (1,1)"""
        X0 = torch.tensor(X0, dtype=torch.float32)
        X0.requires_grad = True
        X1 = torch.tensor(X1, dtype=torch.float32)
        X1.requires_grad = True
        X1hat = torch.zeros_like(X0)
        X1hat.requires_grad = True
        H = torch.tensor(H, dtype=torch.float32)
        H.requires_grad = True
        B = torch.tensor(B, dtype=torch.float32)
        B.requires_grad = True

        if num_meth == "Forward Euler":
            XX_NN = model(X0, H)
            XX_f = XX_NN[0]
            XX_sigma = XX_NN[1]
            X1_hat = X0 + H * XX_f + B * XX_sigma
            #X1_hat = X0 + H*XX_f

            #XX_NN_0 = model(X0, 0 * H)

            # print(list(range(int(X1.shape[1] / N_data))))
            X1_mean = torch.cat(([torch.mean(X1[:, j * N_data: (j + 1) * N_data], dim=1, keepdim=True).repeat([1, N_data]) for j in range(int(X1.shape[1] / N_data))]), dim=1)
            X1_hat_mean = torch.cat(([torch.mean(X1_hat[:, j * N_data: (j + 1) * N_data], dim=1, keepdim=True).repeat([1, N_data]) for j in range(int(X1_hat.shape[1] / N_data))]), dim=1)

            # loss_mean = (((X1_hat - (X0 + H * XX_NN_0[0] + B * XX_NN_0[1]))/H**2).abs() ** 1).mean()
            loss_mean = (((X1_hat - X1) / H ** 2).abs() ** 1).mean()
            loss_sd = ((((X1_hat - X1_hat_mean) ** 2 - (X1 - X1_mean) ** 2) / H ** 2).abs() ** 1).mean()

        if num_meth == "MidPoint":
            XX_f = (model(X0, H)[0] + model(X1, H)[0]) / 2
            XX_sigma = (model(X0, H)[1] + model(X1, H)[1]) / 2
            X1_hat = X0 + H * XX_f + B * XX_sigma

            #print(list(range(int(X1.shape[1] / N_data))))
            X1_mean = torch.cat(([torch.mean(X1[:, j * N_data: (j + 1) * N_data], dim=1, keepdim=True).repeat([1,N_data]) for j in range(int(X1.shape[1]/N_data))]), dim=1)
            X1_hat_mean = torch.cat(([torch.mean(X1_hat[:, j * N_data: (j + 1) * N_data], dim=1, keepdim=True).repeat([1,N_data]) for j in range(int(X1_hat.shape[1]/N_data))]), dim=1)

            #loss_mean = (((X1_hat - (X0 + H * XX_NN_0[0] + B * XX_NN_0[1]))/H**2).abs() ** 1).mean()
            loss_mean = (((X1_hat - X1)/H**3).abs() ** 1).mean()
            loss_sd = ((((X1_hat - X1_hat_mean)**2 - (X1 - X1_mean)**3)/H**2).abs() ** 1).mean()
        #loss_sd = ((((X1_hat)**2 - (X1)**2)/H**2).abs() ** 1).mean()
        #print('Loss[mean]:', format(loss_mean, '.2E'), '-  Loss[std]: =', format(loss_sd, '.2E'))

        loss = loss_mean + loss_sd

        return loss

    def train(self, model, Data):
        """Makes the training on the data
        Inputs:
        - model: Neural network which will be optimized
        - Data: Tuple of tensors - Set of data created
        => Returns the lists Loss_train and Loss_test of the values of the Loss respectively for training and test,
        and best_model, which is the best apporoximation of the modified field computed"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("Training...")
        print(150 * "-")

        X0_train = Data[0]
        X0_test = Data[1]
        X1_train = Data[2]
        X1_test = Data[3]
        H_train = Data[4]
        H_test = Data[5]
        B_train = Data[6]
        B_test = Data[7]

        for W in model.parameters():
            W = torch.randn(size = W.shape)*W.std() + W.mean()*torch.ones_like(W)
        optimizer = optim.AdamW(model.parameters(), lr=alpha, betas=(0.9, 0.999), eps=1e-8, weight_decay=Lambda,amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train = [] # list for loss_train values
        Loss_test = []  # List for loss_test values

        for epoch in range(N_epochs + 1):
            #IDX = (torch.randperm(X0_train.shape[1])[:BS//2], torch.randperm(X0_train.shape[1])[BS//2+1:BS])
            #for ixs in IDX:
            for ixs in torch.split(torch.arange(X0_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                X0_batch = X0_train[:, ixs]
                X1_batch = X1_train[:, ixs]
                H_batch = H_train[:, ixs]
                B_batch = B_train[:, ixs]
                loss_train = self.Loss(X0_batch, X1_batch, H_batch, B_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(X0_test, X1_test, H_test, B_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Step', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        return (Loss_train, Loss_test, best_model)

class Integrate(Train, Data):
    """Class for integration of the SDE"""

    def integrate(self, model, name, save_fig):
        """Prints the values of the Loss along the epochs, trajectories and errors.
        Inputs:
        - model: Best model learned during training, Loss_train and Loss_test
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        def write_size():
            """Changes the size of writings on all windows"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.legend(fontsize=7)
            pass

        def write_size3D():
            """Changes the size of writings on all windows - 3d variant"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            axes.zaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            axes.zaxis.set_tick_params(labelsize=7)
            plt.legend(fontsize=7)
            pass

        start_time_integrate = time.time()

        Model , Loss_train , Loss_test = model[0] , model[1] , model[2]

        print(" ")
        print(100 * "-")
        print("Integration...")
        print(100 * "-")

        fig = plt.figure()

        ax = fig.add_subplot(2, 1, 2)
        plt.plot(range(len(Loss_train)), Loss_train, color='green', label='$Loss_{train}$')
        plt.plot(range(len(Loss_test)), Loss_test, color='red', label='$Loss_{test}$')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.title('Evolution of the Loss function (MLP)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        write_size()

        # Time interval for integration
        TT = np.arange(0, T_simul + h_simul, h_simul)

        # Computation of exact flow and associated brownian motion
        start_time_exact = time.time()
        phi = self.Exact_Flow(X_0=X_0_start(dyn_syst), t_0=0, T=T_simul, h=h_simul)
        print("Integration time of SDE with accurate method (one trajectory - h:min:s):", datetime.timedelta(seconds=time.time() - start_time_exact))
        X_exact, B = phi[0], phi[2]

        # Computation of numerical approximation
        start_time_num = time.time()
        X_num = np.zeros_like(X_exact)
        X_num[:, 0] = X_0_start(dyn_syst)
        for n in range(TT.size-1):
            if num_meth == "Forward Euler":
                X_num[:, n+1] = X_num[:, n] + h_simul * self.f_array(0.0, X_num[:, n]) + (B[n+1] - B[n])*self.sigma_array(0.0, X_num[:, n])
            if num_meth == "MidPoint":
                def F_iter(y):
                    """Iteration function for MidPoint method via fixed-point method.
                    Input:
                    - y: Array of shape (d,) - Space variable."""
                    return X_num[:, n] + h_simul * (Vector_field().f_array(0.0, y) + Vector_field().f_array(0.0, X_num[:, n])) / 2 + (Vector_field().sigma_array(0.0, X_num[:, n]) + Vector_field().sigma_array(0.0, y)) * (B[n+1] - B[n]) / 2
                yy = X_num[:, n]
                for k in range(N_iter):
                    yy = F_iter(yy)
                X_num[:, n + 1] = yy
        print("Integration time of SDE with " + num_meth + " (one trajectory - h:min:s):", datetime.timedelta(seconds=time.time() - start_time_num))

        # Computation of numerical approximation with Modified field [only available for Black & Scholes model and Forward Euler method]
        if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
            start_time_num_modif = time.time()
            X_num_modif = np.zeros_like(X_exact)
            X_num_modif[:, 0] = X_0_start(dyn_syst)
            for n in range(TT.size - 1):
                if num_meth == "Forward Euler":
                    X_num_modif[:, n + 1] = X_num_modif[:, n] + h_simul * self.f_modif_array(0.0, X_num_modif[:, n], h_simul) + (B[n + 1] - B[n]) * self.sigma_modif_array(0.0, X_num_modif[:, n], h_simul)
            print("Integration time of SDE with " + num_meth + " and Modified field (one trajectory - h:min:s):", datetime.timedelta(seconds=time.time() - start_time_num_modif))

        # Computation of numerical approximation with Machine Learning
        start_time_num_theta = time.time()
        X_num_theta = np.zeros_like(X_exact)
        X_num_theta[:, 0] = X_0_start(dyn_syst)
        for n in range(TT.size-1):
            if num_meth == "Forward Euler":
                F_theta = Model(torch.tensor(X_num_theta[:, n]).reshape(d,1), torch.tensor(h_simul).reshape(1,1))
                X_num_theta[:, n+1] = X_num_theta[:, n] + h_simul * F_theta[0].detach().numpy().reshape(d,) + (B[n+1] - B[n])*F_theta[1].detach().numpy().reshape(d,)
                #X_num_theta[:, n+1] = X_num_theta[:, n] + h_simul * F_theta[0].detach().numpy().reshape(d,) + (B[n+1] - B[n])*self.sigma_array(0.0, X_num_theta[:, n])
            if num_meth == "MidPoint":
                def F_iter_theta(y):
                    """Iteration function for MidPoint method via fixed-point method.
                    Input:
                    - y: Array of shape (d,) - Space variable."""
                    F_theta = Model(torch.tensor(X_num_theta[:, n]).reshape(d, 1), torch.tensor(h_simul).reshape(1, 1))
                    F_theta_0 = Model(torch.tensor(y).reshape(d, 1), torch.tensor(h_simul).reshape(1, 1))
                    return X_num_theta[:, n] + h_simul * (F_theta[0].detach().numpy().reshape(d,) + F_theta_0[0].detach().numpy().reshape(d,)) / 2 + (F_theta[1].detach().numpy().reshape(d,) + F_theta_0[1].detach().numpy().reshape(d,)) * (B[n+1] - B[n]) / 2
                yy = X_num_theta[:, n]
                for k in range(N_iter):
                    yy = F_iter_theta(yy)
                X_num_theta[:, n + 1] = yy
        print("Integration time of SDE with "+num_meth+" and Machine Learning (one trajectory - h:min:s):",datetime.timedelta(seconds=time.time() - start_time_num_theta))

        print("   ")
        norm_exact = np.linalg.norm(np.array([np.linalg.norm((X_exact)[:, i]) for i in range((X_exact).shape[1])]) , np.infty) # Norm of the exact solution
        err_num = np.array([np.linalg.norm((X_exact - X_num)[:, i]) for i in range((X_exact - X_num).shape[1])])
        Err_num = np.linalg.norm(err_num, np.infty)/norm_exact
        err_num_theta = np.array([np.linalg.norm((X_exact - X_num_theta)[:, i]) for i in range((X_exact - X_num_theta).shape[1])])
        Err_num_theta = np.linalg.norm(err_num_theta, np.infty) / norm_exact
        if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
            err_num_modif = np.array([np.linalg.norm((X_exact - X_num_modif)[:, i]) for i in range((X_exact - X_num_modif).shape[1])])
            Err_num_modif = np.linalg.norm(err_num_modif, np.infty) / norm_exact
        print("Relative error between exact and numerical flow (one trajectory):", format(Err_num, '.4E'))
        if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
            print("Relative error between exact and numerical flow with Modified field (one trajectory):", format(Err_num_modif, '.4E'))
        print("Relative error between exact and numerical flow with Machine Learning (one trajectory):", format(Err_num_theta, '.4E'))

        if d == 1:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(TT, X_exact[0, :], color='black', linestyle='dashed', label = "Exact flow") #label=r"$\varphi_{t_n}^f(y_0)$")
            plt.plot(TT, X_num[0, :], color='red', label = "Numerical flow") #label="$(\Phi_{h}^{f})^n(y_0)$")
            plt.plot(TT, X_num_theta[0, :], color='green', label = "Numerical flow + ML") #label="$(\Phi_{h}^{f_{\\theta})^n(y_0)$")
            if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
                plt.plot(TT, X_num_theta[0, :], color='cyan', label = "Numerical flow + Modif") # label="$(\Phi_{h}^{f_h})^n(y_0)$")
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
                plt.ylim(min([min(err_num[1:]),min(err_num_modif[1:]),min(err_num_theta[1:])]),max([max(err_num[1:]),max(err_num_modif[1:]),max(err_num_theta[1:])]))
            else:
                plt.ylim(min([min(err_num[1:]),min(err_num_theta[1:])]),max([max(err_num[1:]),max(err_num_theta[1:])]))
            plt.yscale('log')
            plt.plot(TT, err_num, color="blue", label = "Numerical flow") # label="$|$"+r"$\varphi_{t_n}^f(y_0) - (\Phi_{h}^{f})^n(y_0) |$")
            plt.plot(TT, err_num_theta, color="orange", label = "Numerical flow + ML") #label = "Numerical flow + ML") # label="$|$"+r"$\varphi_{t_n}^f(y_0) - (\Phi_{h}^{f_{\theta})^n(y_0)$")
            if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
                plt.plot(TT, err_num_modif, color="magenta",  label = "Numerical flow + Modif") # label="$|$"+r"$\varphi_{t_n}^f(y_0) - (\Phi_{h}^{\Tilde{f}_h})^n(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

        if d == 2:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(X_exact[0, :], X_exact[1, :], color='black', linestyle='dashed',label=r"$\varphi_{t_n}^f(y_0)$")
            plt.plot(X_num[0, :], X_num[1, :], color='red', label="$(\Phi_{h}^{f})^n(y_0)$")
            plt.plot(X_num_theta[0, :], X_num_theta[1, :], color='green', label="$(\Phi_{\\theta,h}^{f})^n(y_0)$")
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min([min(err_num[1:]),min(err_num_theta[1:])]),max([max(err_num[1:]),max(err_num_theta[1:])]))
            plt.yscale('log')
            plt.plot(TT, err_num, color="blue", label="$|$"+r"$\varphi_{t_n}^f(y_0) - (\Phi_{h}^{f})^n(y_0) |$")
            plt.plot(TT, err_num_theta, color="orange", label="$|$"+r"$\varphi_{t_n}^f(y_0) - (\Phi_{\theta,h}^{f})^n(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()



        #plt.show()
        f = plt.gcf()
        dpi = f.get_dpi()
        h, w = f.get_size_inches()
        f.set_size_inches(h * 1.7, w * 1.7)

        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()

        print("Computation time for integration (h:min:s):",
              str(datetime.timedelta(seconds=int(time.time() - start_time_integrate))))

        pass

class Trajectories(Integrate):
    """Class for the study of convergence of trajectories"""

    def traj(self, model, name, save_fig):
        """Prints the global errors according to the step of the numerical method
        Inputs:
        - model: Model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        HH = np.exp(np.linspace(np.log(step_h[0]) , np.log(step_h[1]) , 9 ))
        ERR_num = np.zeros(np.size(HH))
        ERR_num_theta = np.zeros(np.size(HH))
        if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
            ERR_num_modif = np.zeros(np.size(HH))

        Model = model[0]

        N_traj = 10000 # Number of computed trajectories in order to get expected value (LLN).

        TT_ref = np.sort(np.concatenate([np.arange(0,T_simul+HH[j],HH[j]) for j in range(np.size(HH))]))
        TT_ref = np.unique(np.round(TT_ref, decimals=12), equal_nan=False)

        # Computation of exact solutions (very accurate approximation - references)
        print(" > Computation of reference solutions")
        X_exact_ref = np.zeros((d, np.size(TT_ref), N_traj))
        B_ref = np.zeros((np.size(TT_ref), N_traj))
        for n_traj in range(N_traj):
            char = "       - Trajectory "+ str(n_traj + 1) + "/" + str(N_traj) + " "
            print(char, end = "\r")
            X_exact_ref[:, 0, n_traj] = X_0_start(dyn_syst)
            n_it = 20
            for n in range(np.size(TT_ref)-1):
                delta_t = (TT_ref[n+1]-TT_ref[n])/n_it
                B_ref[n + 1, n_traj] = B_ref[n, n_traj]
                XX = X_exact_ref[:, n, n_traj]
                if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
                    BB = np.random.normal(0, (TT_ref[n+1]-TT_ref[n]) ** 0.5)
                    X_exact_ref[:, n + 1, n_traj] = np.exp( (Mu - Sigma ** 2 / 2) * (TT_ref[n+1]-TT_ref[n]) + Sigma * BB) * X_exact_ref[:, n, n_traj]
                    B_ref[n + 1, n_traj] = B_ref[n + 1, n_traj] + BB
                else:
                    for j in range(n_it):
                        BB = np.random.normal(0, delta_t ** 0.5)
                        B_ref[n + 1, n_traj] = B_ref[n + 1, n_traj] + BB
                        if num_meth == "Forward Euler":
                            XX = XX + delta_t * self.f_array(0.0, XX) + self.sigma_array(0.0, XX) * BB
                        if num_meth == "MidPoint":
                            def F_iter(y):
                                """Iteration function for MidPoint method via fixed-point method.
                                Input:
                                - y: Array of shape (d,) - Space variable"""
                                return XX + delta_t * (Vector_field().f_array(0.0, y) + Vector_field().f_array(0.0, XX)) / 2 + (Vector_field().sigma_array(0.0, XX) + Vector_field().sigma_array(0.0, y)) * BB / 2
                            xx = XX
                            for k in range(N_iter):
                                xx = F_iter(xx)
                            XX = xx

                    X_exact_ref[:, n + 1, n_traj] = XX

        # Computation of numerical approximations (without and with Machine Learning)
        print(" ")
        print(" > Computation of numerical approximations")
        for jh in range(np.size(HH)):
            hh = HH[jh]

            TT = np.arange(0, T_simul + hh, hh)
            idx = np.zeros_like(TT)
            tol = 1e-12
            for j in range(np.size(idx)):
                k = 0
                while np.abs(TT[j] - TT_ref[k]) > tol:
                    k += 1
                idx[j] = k
            idx = np.array(idx, dtype=int)

            X_exact = X_exact_ref[:, idx, :]
            B = B_ref[idx, :]
            X_num, X_num_theta = np.zeros_like(X_exact), np.zeros_like(X_exact)
            if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
                X_num_modif = np.zeros_like(X_exact)

            for n_traj in range(N_traj):
                char = "       - h = "+str(format(hh, '.4E')) + " - Trajectory " + str(n_traj + 1) + "/" + str(N_traj) + "   "
                print(char,end="\r")

                X_num[:, 0, n_traj], X_num_theta[:, 0, n_traj] = X_0_start(dyn_syst), X_0_start(dyn_syst)
                for n in range(np.size(TT)-1):
                    if num_meth == "Forward Euler":
                        X_num[:, n + 1, n_traj] = X_num[:, n, n_traj] + hh * self.f_array(0.0, X_num[:, n, n_traj]) + (B[n + 1, n_traj] - B[n, n_traj]) * self.sigma_array(0.0, X_num[:, n, n_traj])
                        F_theta = Model(torch.tensor(X_num_theta[:, n, n_traj]).reshape(d, 1), torch.tensor(hh).reshape(1, 1).float())
                        X_num_theta[:, n + 1, n_traj] = X_num_theta[:, n, n_traj] + hh * F_theta[0].detach().numpy().reshape(d, ) + (B[n + 1, n_traj] - B[n, n_traj]) * F_theta[1].detach().numpy().reshape(d, )
                    if num_meth == "MidPoint":
                        def F_iter(y):
                            """Iteration function for MidPoint method via fixed-point method.
                            Input:
                            - y: Array of shape (d,) - Space variable."""
                            return X_num[:, n, n_traj] + hh * (Vector_field().f_array(0.0, y) + Vector_field().f_array(0.0, X_num[:, n, n_traj])) / 2 + (Vector_field().sigma_array(0.0, X_num[:, n, n_traj]) + Vector_field().sigma_array(0.0, y)) * (B[n + 1, n_traj] - B[n, n_traj]) / 2

                        def F_iter_theta(y):
                            """Iteration function for MidPoint method via fixed-point method.
                            Input:
                            - y: Array of shape (d,) - Space variable."""
                            F_theta = Model(torch.tensor(X_num_theta[:, n, n_traj]).reshape(d, 1), torch.tensor(hh).reshape(1, 1).float())
                            F_theta_0 = Model(torch.tensor(y).reshape(d, 1), torch.tensor(hh).reshape(1, 1).float())
                            return X_num_theta[:, n, n_traj] + hh * (F_theta[0].detach().numpy().reshape(d,) + F_theta_0[0].detach().numpy().reshape(d,)) / 2 + (F_theta[1].detach().numpy().reshape(d,) + F_theta_0[1].detach().numpy().reshape(d,)) * (B[n + 1, n_traj] - B[n, n_traj]) / 2

                        yy_num, yy_num_theta = X_num[:, n, n_traj], X_num_theta[:, n, n_traj]

                        for k in range(N_iter):
                            yy_num = F_iter(yy_num)
                            yy_num_theta = F_iter_theta(yy_num_theta)
                        X_num[:, n + 1, n_traj], X_num_theta[:, n + 1, n_traj] = yy_num, yy_num_theta
                if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
                    X_num_modif[:, 0, n_traj] = X_0_start(dyn_syst)
                    for n in range(np.size(TT) - 1):
                        X_num_modif[:, n + 1, n_traj] = X_num_modif[:, n, n_traj] + hh * self.f_modif_array(0.0, X_num_modif[:, n, n_traj], hh) + (B[n + 1, n_traj] - B[n, n_traj]) * self.sigma_modif_array(0.0, X_num_modif[:, n, n_traj], hh)


            # Computation of the norms of the exact solutions for the computation of relative errors
            #norm_exact = np.linalg.norm(np.linalg.norm(X_exact, ord=2, axis=0), ord=np.infty, axis=0)
            norm_exact = np.linalg.norm(np.linalg.norm(np.mean(X_exact, axis=2), ord=2, axis=0), ord=np.infty, axis=0)
            #err_num = np.mean(np.linalg.norm(np.linalg.norm(X_exact - X_num, ord=2, axis=0), ord=np.infty, axis=0)/norm_exact, axis=0)
            err_num = np.linalg.norm(np.linalg.norm(np.mean(X_exact - X_num, axis=2), ord=2, axis=0), ord=np.infty, axis=0) / norm_exact
            err_num_theta = np.linalg.norm(np.linalg.norm(np.mean(X_exact - X_num_theta, axis=2), ord=2, axis=0), ord=np.infty, axis=0) / norm_exact
            ERR_num[jh], ERR_num_theta[jh] = err_num, err_num_theta
            if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
                err_num_modif = np.linalg.norm(np.linalg.norm(np.mean(X_exact - X_num_modif, axis=2), ord=2, axis=0), ord=np.infty, axis=0) / norm_exact
                ERR_num_modif[jh] = err_num_modif
        plt.figure()
        plt.title("Error between trajectories")
        plt.plot(HH, ERR_num, linestyle="dashed" , marker="s", color="red" , label = "Numerical error")
        plt.plot(HH, ERR_num_theta, linestyle="dashed" , marker="s", color="green" , label = "Numerical error + ML")
        if dyn_syst == "Black_Scholes" and num_meth == "Forward Euler":
            plt.plot(HH, ERR_num_modif, linestyle="dashed", marker="s", color="cyan", label="Numerical error + Modif")
        plt.legend()
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Step size $h$")
        plt.ylabel("Global error")
        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()
        pass


def ExData(name_data="DataSDE"):
    """Creates data X0, X1 with the function Exact_Flow
    Input:
    - name_data: Character string - Name of the registered tuple containing the data (default: "DataSDE")"""
    DataSDE = Data().Data_create(K_data)
    torch.save(DataSDE, name_data)
    pass

def ExTrain(name_model='model_SDE', name_data='DataSDE'):
    """Launches training and computes Loss_train, loss_test and best_model with the function Train().train
    Saves the files Loss_train, Loss_test and best_model with a given name
    Inputs (character strings):
    - name_model: Name of the file saved for best_model (default: "model_SDE")
    - name_data: Name of the file containing the created data (default: "DataSDE") used for training"""
    DataSDE = torch.load(name_data)
    Loss_train, Loss_test, best_model = Train().train(model=NN(), Data=DataSDE)
    torch.save((best_model,Loss_train,Loss_test), name_model)
    pass

def ExIntegrate(name_model="model_SDE", name_graph="Simulation_SDE_ML", save=False):
    """Launches integration of the main equation and modified equation with the chosen model
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with learned modified fields, and Loss_train/Loss_test. Default: model_SDE
    - name_graph: Character string - Name of the graph which will be registered. Default: Simulation_SDE_ML
    - save: Boolean - Saves the figure or not. Default: False"""
    Lmodel = torch.load(name_model)
    Integrate().integrate(model=Lmodel, name=name_graph,save_fig=save)
    pass

def ExTraj(name_model="model_SDE", name_graph="Simulation_Convergence_SDE_ML", save=False):
    """plots the curves of convergence between the trajectories integrated with f and F_app with the numerical method chosen
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with learned modified fields, and Loss_train/Loss_test. Default: model_SDE
    - name_graph: Character string - Name of the graph which will be registered. Default: Simulation_Convergence_SDE_ML
    - save: Boolean - Saves the figure or not. Default: False"""
    Lmodel = torch.load(name_model)
    Trajectories().traj(model=Lmodel, name=name_graph, save_fig=save)
    pass