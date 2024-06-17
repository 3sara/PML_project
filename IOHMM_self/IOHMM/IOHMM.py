import numpy as np, scipy as sp
from scipy.special import logsumexp
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

torch.autograd.set_detect_anomaly(True)
#test
# assuming a linear regression emission model and a logistic regression transition model

class IOHMM_model:
    def __init__(self, num_states, inputs, outputs, max_iter, tol, initial_pi=None, transition_matrix=None, emission_matrix=None, log_sd=None):
        self.num_states = num_states
        self.inputs = inputs
        self.outputs = outputs
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

        # self.initial_pi = torch.ones(num_states) / num_states
        # transition matrix: the coefficients for a logistic regression model, size is (num_states, num_states, inputs.shape[0])
        # first dimension is the current state, second dimension is the next state, third dimension is the input
        # self.transition_matrix = 2 * torch.rand(num_states, num_states, inputs.shape[1], dtype=torch.float32) -1
        # emission matrix: the coefficients for a linear regression model, size is (num_states, outputs.shape[0])
        # self.emission_matrix = 2 * torch.rand(num_states, inputs.shape[1], dtype=torch.float32) -1

        if initial_pi is not None:
            # self.initial_pi = initial_pi
            self.initial_pi = nn.Parameter(initial_pi, requires_grad=True)
        else:
            self.initial_pi = nn.Parameter(torch.log(torch.ones(num_states)/num_states), requires_grad=True)
            # self.initial_pi = torch.ones(num_states) / num_states
        if transition_matrix is not None:
            self.transition_matrix = nn.Parameter(transition_matrix, requires_grad=True)
        else:
            self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states, inputs.shape[1] + 1), requires_grad=True)
        if emission_matrix is not None:
            self.emission_matrix = nn.Parameter(emission_matrix, requires_grad=True)
        else:
            self.emission_matrix = nn.Parameter(torch.randn(num_states, inputs.shape[1] + 1), requires_grad=True)
        if log_sd is not None:
            self.log_sd = nn.Parameter(log_sd, requires_grad=True)
        else:
            # setting a higher log_sd was fundamental to make the model stable, otherwise the emission prob would be zero
            self.log_sd = nn.Parameter(torch.log(5.0*torch.ones(num_states)), requires_grad=True)


    def log_softmax_(self, input, state):
        with torch.no_grad():
            # I think is dim = 0
            return (self.transition_matrix[state,:,:] @ torch.cat((torch.tensor([1.0]),input))) - torch.logsumexp(torch.exp(self.transition_matrix[state,:,:] @ torch.cat((torch.tensor([1.0]),input))), dim =0)
                
    def log_softmax(self, input):
        with torch.no_grad():
        
            # Concatenate 1 to the beginning of the input
            input_with_bias = torch.cat((torch.tensor([1.0]), input))

            # Compute transition probabilities for all states
            transition_logits = self.transition_matrix @ input_with_bias

            # Compute softmax probabilities
            #exp_logits = torch.exp(transition_logits)
            #softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

            # Compute log softmax probabilities
            log_softmax_probs = transition_logits - torch.logsumexp(transition_logits, dim=1)
            return log_softmax_probs

    def log_dnorm(self, x, mean, log_sd):
        with torch.no_grad():        
            return (-0.5 * ((x - mean) / torch.exp(log_sd)) ** 2) - (log_sd + torch.log(torch.sqrt(torch.tensor(2 * np.pi))))

    def _forward(self):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            alpha = torch.zeros((T, N))

            # Initialize base cases (t == 0)
            input_with_bias = torch.cat((torch.tensor([1.0]), U[0]))
            log_emission_prob = self.log_dnorm(self.outputs[0], self.emission_matrix @ input_with_bias, self.log_sd)
            alpha[0] = self.initial_pi + log_emission_prob
            alpha[0] -= torch.logsumexp(alpha[0], dim = 0)

            # Compute forward probabilities (t > 0)
            for t in range(1, T):
                input_with_bias = torch.cat((torch.tensor([1.0]), U[t]))
                log_emission_prob = self.log_dnorm(self.outputs[t], self.emission_matrix @ input_with_bias, self.log_sd)

                log_transition_prob = self.log_softmax(U[t])
                # sum on the columns of transition_prob, quite sure about axis = 0
                log_transition_prob += alpha[t-1].unsqueeze(1)

                alpha[t] = log_emission_prob + torch.logsumexp(log_transition_prob, axis=0)
                # normalize
                alpha[t] -= torch.logsumexp(alpha[t], dim=0)

            return alpha


    def _backward(self):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            beta = torch.zeros((T, N))

            # Initialize base cases (t == T-1)
            # parallel version
            input_with_bias = torch.cat((torch.tensor([1.0]), U[-1]))
            log_emission_prob = self.log_dnorm(self.outputs[-1], self.emission_matrix @ input_with_bias, self.log_sd)
            # to normalize
            beta[-1] = log_emission_prob
            beta[-1] -= torch.logsumexp(beta[-1],dim=0)

            # Compute forward probabilities (t > 0)
            for t in reversed(range(T-1)):
                input_with_bias = torch.cat((torch.tensor([1.0]), U[t]))
                log_emission_prob = self.log_dnorm(self.outputs[t], self.emission_matrix @ input_with_bias, self.log_sd)
                
                log_transition_prob = self.log_softmax(U[t])
                log_transition_prob += beta[t+1].unsqueeze(1)
                
                beta[t] = log_emission_prob + torch.logsumexp(log_transition_prob, axis=0)
                # normlize
                beta[t] -= torch.logsumexp(beta[t],dim=0)

            return beta


    def _compute_gamma(self, alpha=None, beta=None):
        with torch.no_grad():
            # in the paper it says that zeta_it = P(x_t=i | all obs of inputs from 0 to t)
            # so it should simply be alpha, but not sure
            gamma = alpha + beta
            # normalize
            gamma -= torch.logsumexp(gamma, axis=1).reshape(-1, 1)
            return gamma


    def _compute_xi(self, alpha=None, beta=None):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            xi = torch.zeros((T, N, N))

            for t in range(0, T):
                transition_prob = self.log_softmax(U[t]).T
                # transpose to maintain the same notation of the article, in the row next state, in the columns previous state
                #print(f"transition prob: {transition_prob}")
                #print(f"alpha[t-1]: {alpha[t-1]}")
                #print(f"beta[t]: {beta[t]}")
                # not sure if i have to unsqueeze alpha or beta, is it the same?
                xi[t, :, :] = transition_prob + beta[t].unsqueeze(1) + alpha[t-1]

            # S = torch.sum(alpha[T-1, :], axis=0)
            # xi /= S

            # normalize
            # to check xi
            a = torch.logsumexp(xi, axis=1)
            #a = torch.sum(a, axis=1)
            xi -= a[:, None]
            return xi
    
    def _log_likelihood(self, gamma, xi):
        likelihood = 0
        likelihood += torch.sum(torch.exp(gamma[0]) * self.initial_pi)
        
        for t, output in enumerate(self.outputs):
            
            # first term of the sum

            x = self.emission_matrix @ (torch.cat((torch.tensor([1.0]), self.inputs[t])))
            mu = output

            dnorm = (-0.5 * (((mu - x) / torch.exp(self.log_sd)) ** 2)) - ((self.log_sd) +torch.log(torch.sqrt(torch.tensor(2 * np.pi))))

            likelihood += torch.sum(torch.exp(gamma[t, :]) * dnorm)
            # second term of the sum

            # Concatenate 1 to the beginning of the input
            input_with_bias = torch.cat((torch.tensor([1.0]), self.inputs[t]))
            # Compute transition probabilities for all states
            transition_logits = self.transition_matrix @ input_with_bias
            # Compute softmax probabilities
            #exp_logits = torch.exp(transition_logits)
            #softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)


            #compute directly log softmax probabilities
            softmax_probs = transition_logits - torch.logsumexp(transition_logits, dim=1, keepdim=True)
            # to check relation between xi and softmax
            likelihood += torch.sum(torch.exp(xi[t, :, :]) * softmax_probs)
                
        return likelihood



    def _baum_welch(self):
        # sgd not good, better LBFGS
        optimizer = optim.SGD([self.initial_pi, self.transition_matrix, self.emission_matrix, self.log_sd], lr=0.1)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)  # Decrease lr by 0.9 every 1 steps
        
        old_log_likelihood = -torch.inf

        for i in range(self.max_iter):
            # E-step: Compute the posterior probabilities

            alpha = self._forward()
            beta = self._backward()
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(alpha, beta)

            # M-step
            def closure():
                optimizer.zero_grad()
                loss = -self._log_likelihood(gamma, xi)
                loss.backward()
                return loss
            print(f"Iteration {i+1}, likelihood: {old_log_likelihood}")

            optimizer.step(closure)
            #scheduler.step()
            

            with torch.no_grad():
                new_log_likelihood = self._log_likelihood(gamma, xi)
                self.history.append(new_log_likelihood)
                # Check for convergence
                if torch.abs(new_log_likelihood - old_log_likelihood) < self.tol:
                    print("convergence reached :)")
                    print(f"final likelihood: {new_log_likelihood}")
                    break
                old_log_likelihood = new_log_likelihood

        if i == self.max_iter:    
            print("convergence not reached")

    def viterbi(self):
        with torch.no_grad():
            U = self.inputs

            alpha = self._forward()
            path = []
            last_state = torch.argmax(alpha[-1])
            path.append(last_state.item())

            for i in reversed(range(len(self.outputs) - 1)):
                transition_prob = self.log_softmax_(U[i + 1], last_state)
                last_state = torch.argmax(torch.log(transition_prob) + alpha[i])
                path.append(last_state.item())

            path.reverse()
            return path
    

    def predict(self, u_t1):
        """
        Output prediction is defined as eta_t+1 . In particular
            phi_ij_t+1 = P(X_t+1 = i | X_t = j, u_t+1)
            phi_j_t+1 = softmax(theta_j @ u_t+1)
            rho_i_t+1 = P(X_t+1 = i | u_1, ..., u_t+1) = sum_j(phi_ij_t+1 * rho_j_t)
            eta_i_t = E[Y_t+1 | X_t+1 = i, u_t+1] = delta_i @ u_t+1
            eta_t+1 = sum_i(rho_i_t+1 * eta_i_t+1)
        
        The theta_j are the transition matrix, the delta_i are the emission matrix.
        """


        # Compute rho_i_t
        rho_t = self._compute_gamma(self._forward(), self._backward())[-1]

        # Compute rho_i_t+1
        rho_t1 = torch.sum(self.softmax(u_t1) * rho_t, axis=1)
        
        # Compute eta_i_t
        eta_i_t1 = self.emission_matrix @ torch.cat((torch.tensor([1.0]), u_t1))

        # Compute eta_t+1
        eta_t1 = torch.sum(rho_t1 * eta_i_t1)

        return eta_t1