import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

class IOHMM_model:
    def __init__(self, num_states, inputs, outputs, max_iter, tol,
                 log_initial_pi=None, theta_transition=None, theta_emission=None, log_sd=None):
        
        self.num_states = num_states
        self.inputs = inputs
        self.outputs = outputs
        self.T = inputs.shape[0]
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

        """
        Parameters:
            num_states: int
                the number of hidden states
            
            inputs: torch.tensor
                the input data, size is (num_samples, num_features)

            outputs: torch.tensor
                the output data, size is (num_samples)

            max_iter: int
                the maximum number of iterations for the EM algorithm
            
            tol: float
                the tolerance for the convergence of the EM algorithm
            
            log_initial_pi: torch.tensor
                the initial log probabilities of the initial states, size is (num_states)
                uniform distribution by default
            
            theta_transition: torch.tensor
                coefficients of the transition model, size is (num_states, num_states, num_features + 1)
                random initialization by default
            
            theta_emission: torch.tensor
                coefficients of the emission model, size is (num_states, num_features + 1)
                random initialization by default
            
            log_sd: torch.tensor
                the log standard deviations of the observation noise, size is (num_states)
                5.0 by default
            
        """

        if log_initial_pi is not None:
            self.log_initial_pi = nn.Parameter(log_initial_pi, requires_grad=True)
        else:
            self.log_initial_pi = nn.Parameter(torch.log(torch.ones(num_states)/num_states), requires_grad=True)

        if theta_transition is not None:
            self.theta_transition = nn.Parameter(theta_transition, requires_grad=True)
        else:
            self.theta_transition = nn.Parameter(torch.randn(num_states, num_states, inputs.shape[1] + 1), requires_grad=True)
        
        if theta_emission is not None:
            self.theta_emission = nn.Parameter(theta_emission, requires_grad=True)
        else:
            self.theta_emission = nn.Parameter(torch.randn(num_states, inputs.shape[1] + 1), requires_grad=True)
        
        if log_sd is not None:
            self.log_sd = nn.Parameter(log_sd, requires_grad=True)
        else:
            self.log_sd = nn.Parameter(torch.log(5.0*torch.ones(num_states)), requires_grad=True)

    
    def state_subnetwork(self, u_t):
        """
        Parameters:
            u_t: torch.tensor, size (num_features)
                the input data point, size is (num_features)

        Returns:
            log_phi: torch.tensor, size (num_states, num_states)
            phi_ij = p(x_t = i | x_{t-1} = j, u_t)
        """
        #with torch.no_grad():

        bias = torch.tensor([1.0])
        input_w_bias = torch.cat((bias, u_t))

        x = torch.matmul(self.theta_transition, input_w_bias)
        log_phi = x - torch.logsumexp(x, dim=0)
        
        return log_phi

    def emission_subnetwork(self, u_t):
        """
        Parameters:
            u_t: torch.tensor, size (num_features)
                the input data point

        Returns:
            eta: torch.tensor, size (num_states)
        """
        #with torch.no_grad():

        bias = torch.tensor([1.0])
        input_w_bias = torch.cat((bias, u_t))

        eta = torch.matmul(self.theta_emission, input_w_bias)
        
        return eta
    

    def log_dnorm(self, x, mean, log_sd):
        """
        Parameters:
            x: torch.tensor 
                the data point
            
            mean: torch.tensor
                the mean of the normal distribution
            
            log_sd: torch.tensor
                the log standard deviation of the normal distribution
        
        Returns:
            log_prob: torch.tensor
                the log probability of the data point
        
        """
        #with torch.no_grad():
            
        return (-0.5 * ((x - mean) / torch.exp(log_sd)) ** 2) - (log_sd + torch.log(torch.sqrt(torch.tensor(2 * np.pi))))
    

    def forward(self):
        """
        Perform the forward pass of the IOHMM model.
        
        Returns:
            log_alpha: torch.tensor, size (num_states, T)
        """
        #with torch.no_grad():

        log_alpha = torch.zeros(self.num_states, self.T)

        # Initialization
        log_alpha_start = self.log_initial_pi

        for i in range(self.num_states):
            mean = self.emission_subnetwork(self.inputs[0])
            log_dnorm_prob = self.log_dnorm(self.outputs[0], mean[i], self.log_sd[i])
            log_phi = self.state_subnetwork(self.inputs[0])
            log_alpha[i, 0] = log_dnorm_prob + torch.logsumexp(log_phi[i, :] + log_alpha_start, dim=0)   

        # Iteration
        for t in range(1, self.T):
            for i in range(self.num_states):
                mean = self.emission_subnetwork(self.inputs[t])
                log_dnorm_prob = self.log_dnorm(self.outputs[t], mean[i], self.log_sd[i])
                log_phi = self.state_subnetwork(self.inputs[t])
                log_alpha[i, t] = log_dnorm_prob + torch.logsumexp(log_phi[i, :]+log_alpha[:, t-1], dim=0)  
        
        return log_alpha
        
    
    def backward(self):
        """
        Perform the backward pass of the IOHMM model.
        
        Returns:
            log_backward: torch.tensor, size (num_states, T)
        """
        #with torch.no_grad():
        log_beta = torch.zeros(self.num_states, self.T)

        # Initialization
        log_beta[:, -1] = torch.zeros(self.num_states)

        # Iteration
        
        for t in range(self.T-2, -1, -1):
            for i in range(self.num_states):
                mean = self.emission_subnetwork(self.inputs[t+1])
                log_dnorm_prob = self.log_dnorm(self.outputs[t+1], mean, self.log_sd)
                log_phi = self.state_subnetwork(self.inputs[t+1])
                log_beta[i, t] = torch.logsumexp(log_phi[:, i] + log_dnorm_prob + log_beta[:, t+1], dim=0)
        
        return log_beta
        
    
    def compute_log_g(self,log_alpha, log_beta):
        """
        Compute the intermediate variable g for the EM algorithm.
        
        Returns:
            g: torch.tensor, size (num_states, T+1)
        """
        #with torch.no_grad():
        log_L = torch.logsumexp(log_alpha[:, -1], dim=0)
        
        g = log_alpha + log_beta - log_L

        return g
    
    def compute_log_h(self, log_alpha, log_beta):
        """
        Compute the intermediate variable h for the EM algorithm.

        Returns:
            h: torch.tensor, size (num_states, num_states, T)
        """
        #with torch.no_grad():
        log_h = torch.zeros(self.num_states, self.num_states, self.T)
        L = torch.logsumexp(log_alpha[:, -1], dim=0)

        for i in range(self.num_states):
            for j in range(self.num_states):
                mean = self.emission_subnetwork(self.inputs[0])
                log_dnorm_prob = self.log_dnorm(self.outputs[0], mean[i], self.log_sd[i])
                log_phi = self.state_subnetwork(self.inputs[0])
                log_h[i, j, 0] = log_dnorm_prob + self.log_initial_pi[j] + log_phi[i, j] + log_beta[i, 0] - L

        for t in range(1,self.T):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    mean = self.emission_subnetwork(self.inputs[t])
                    log_dnorm_prob = self.log_dnorm(self.outputs[t], mean[i], self.log_sd[i])
                    log_phi = self.state_subnetwork(self.inputs[t])
                    log_h[i, j, t] = log_dnorm_prob + log_alpha[j, t-1] + log_phi[i, j] + log_beta[i, t] - L

        return log_h
        
    
    def log_likelihood(self, log_g, log_h):
        """
        Compute the log likelihood of the IOHMM model.
        
        Returns:
            log_likelihood: torch.tensor
        """
        
        l = torch.zeros(1,1)

        for t in range(self.T):
            for i in range(self.num_states):
                mean = self.emission_subnetwork(self.inputs[t])
                log_dnorm_prob = self.log_dnorm(self.outputs[t], mean[i], self.log_sd[i])
                l += torch.exp(log_g[i,t]) * log_dnorm_prob
                for j in range(self.num_states):
                    log_phi = self.state_subnetwork(self.inputs[t])
                    l += torch.exp(log_h[i,j,t]) * log_phi[i,j] 
        
        return l
    

    def log_normalization(self, x):
        return x - torch.logsumexp(x, dim=0)
    
    def viterbi(self):

        prob=torch.zeros((len(self.outputs),self.num_states), dtype=torch.float64)
        path=torch.zeros((len(self.outputs),self.num_states), dtype=torch.int64)
        
        #initialize the first state
        for i in range(self.num_states):
            mean = self.emission_subnetwork(self.inputs[0])
            log_dnorm_prob = self.log_dnorm(self.outputs[0], mean[0], self.log_sd[0])

            prob[0,i] = self.log_normalization(self.log_initial_pi)[i] + log_dnorm_prob

        #complete the matrixes
        for t in range(1,len(self.outputs)):
            for i in range(self.num_states):
                mean = self.emission_subnetwork(self.inputs[t])
                log_dnorm_prob = self.log_dnorm(self.outputs[t], mean[i], self.log_sd[i])
                log_phi = self.state_subnetwork(self.inputs[0])
                prob[t,i] = torch.max(prob[t-1,:] + log_phi + log_dnorm_prob)
                path[t,i]=torch.argmax(prob[t-1,:] + log_phi + log_dnorm_prob)
        
        state_sequence=torch.zeros(len(self.outputs), dtype=torch.int64)
        state_sequence[-1]=torch.argmax(prob[-1,:])
        for t in range(len(self.outputs)-2,-1,-1):
            state_sequence[t]=path[t+1,state_sequence[t+1]]
        return state_sequence
    
    def viterbi_(self):
        with torch.no_grad():
            U = self.inputs
            log_alpha = self.forward()
            
            path = []
            last_state = torch.argmax(log_alpha[-1])
            path.append(last_state.item())

            for t in reversed(range(len(self.outputs) - 1)):
                transition_prob = self.state_subnetwork(U[t + 1])[last_state]

                last_state = torch.argmax(transition_prob + log_alpha[:,t])
                path.append(last_state.item())

            path.reverse()
            return path
        

    
    def baum_welch(self):

        optimizer = optim.SGD([self.log_initial_pi, self.theta_transition, self.theta_emission, self.log_sd], lr=0.2)
        
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)  # Decrease lr by 0.9 every 1 steps
        
        old_log_likelihood = -torch.inf

        for i in range(self.max_iter):
            print(f"iteration {i}\r")
            print(f"old likelihood: {old_log_likelihood}")
            self.history.append(old_log_likelihood)

            # E-step
            log_alpha = self.forward()
            log_beta = self.backward()
            log_g = self.compute_log_g(log_alpha, log_beta)
            log_h = self.compute_log_h(log_alpha,log_beta)

            # M-step
            def closure():
                optimizer.zero_grad()
                loss = -self.log_likelihood(log_g, log_h)
                loss.backward()
                return loss

            optimizer.step(closure)
            #scheduler.step()

            # Check for convergence
            with torch.no_grad():

                new_log_likelihood = self.log_likelihood(log_g, log_h)
                
                if torch.abs(new_log_likelihood - old_log_likelihood) < self.tol:
                    print("convergence reached :)")
                    print(f"final likelihood: {new_log_likelihood}")
                    break

                old_log_likelihood = new_log_likelihood

    def plot_state_distribution(self):
        with torch.no_grad():
            for state in range(self.num_states):
                print(f"State {state} distribution:")
                mean = self.emission_subnetwork(self.inputs[0])
                #log_dnorm_prob = self.log_dnorm(x, mean[state], self.log_sd[state])
                print(f"Mean: {mean[state]}")
                print(f"Standard deviation: {torch.exp(self.log_sd[state])}")
                print("\n")
                #plot di una normale con media mean e standard deviation exp(log_sd[state])
                x = np.linspace(-10, 10, 100)
                y = np.exp(self.log_dnorm(torch.tensor(x), mean[state], self.log_sd[state]))
                plt.plot(x, y)
                plt.show()


            
    

        





                
