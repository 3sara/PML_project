import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

class IOHMM_model:
    def __init__(self, num_states, inputs, outputs, max_iter, tol):
        self.num_states = num_states
        self.inputs = inputs
        self.outputs = outputs
        self.max_iter = max_iter
        self.tol = tol

        self.initial_pi = nn.Parameter(torch.ones(num_states) / num_states, requires_grad=True)
        self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states, inputs.shape[1] + 1), requires_grad=True)
        # self.transition_matrix = nn.Parameter(torch.ones(num_states, num_states, inputs.shape[1] + 1), requires_grad=True) / num_states
        self.emission_matrix = nn.Parameter(torch.randn(num_states, inputs.shape[1] + 1), requires_grad=True)
        # self.emission_matrix = nn.Parameter(torch.ones(num_states, inputs.shape[1] + 1), requires_grad=True) / num_states
        self.sd = nn.Parameter(torch.ones(num_states), requires_grad=True)

    def softmax(self, input, state):
        with torch.no_grad():
            return torch.exp(self.transition_matrix[state, :, :] @ torch.cat((torch.tensor([1.0]), input))) / torch.sum(torch.exp(self.transition_matrix[state, :, :] @ torch.cat((torch.tensor([1.0]), input))))

    def dnorm(self, x, mean, sd):
        with torch.no_grad():
            return torch.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * torch.sqrt(torch.tensor(2 * np.pi)))

    def _forward(self):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            alpha = torch.zeros((T, N))

            for i in range(N):
                emission_prob = self.dnorm(self.outputs[0], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), U[0]))), self.sd[i])
                alpha[0, i] = self.initial_pi[i] * emission_prob
            alpha[0] /= torch.sum(alpha[0])

            for t in range(1, T):
                for j in range(N):
                    sum = 0
                    emission_prob = self.dnorm(self.outputs[t], self.emission_matrix[j].dot(torch.cat((torch.tensor([1.0]), U[t]))), self.sd[j])
                    for i in range(N):
                        transition_prob = self.softmax(U[t], i)
                        sum += alpha[t - 1, i] * transition_prob[j]
                    alpha[t, j] = sum * emission_prob
                alpha[t] /= torch.sum(alpha[t])

            return alpha

    def _backward(self):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            beta = torch.zeros((T, N))

            for i in range(N):
                emission_prob = self.dnorm(self.outputs[-1], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), U[-1]))), self.sd[i])
                beta[-1, i] = emission_prob
            beta[-1] /= torch.sum(beta[-1])

            for t in reversed(range(T - 1)):
                for j in range(N):
                    sum = 0
                    emission_prob = self.dnorm(self.outputs[t], self.emission_matrix[j].dot(torch.cat((torch.tensor([1.0]), U[t]))), self.sd[j])
                    for i in range(N):
                        transition_prob = self.softmax(U[t], i)
                        sum += beta[t + 1, i] * transition_prob[j]
                    beta[t, j] = sum * emission_prob
                beta[t] /= torch.sum(beta[t])

            return beta

    def _compute_gamma(self, alpha=None, beta=None):
        with torch.no_grad():
            gamma = alpha * beta
            gamma /= torch.sum(gamma, axis=1).reshape(-1, 1)
            return gamma

    def _compute_xi(self, alpha=None, beta=None):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            xi = torch.zeros((T, N, N))

            for t in range(0, T):
                for i in range(N):
                    for j in range(N):
                        transition_prob = self.softmax(U[t], j)
                        xi[t, i, j] = (beta[t, i] * alpha[t - 1, j] * transition_prob[i]) 
                        # in the paper divides by torch.sum(alpha[T - 1]), but this shlould sum to one(?)
            # normalize
            a = torch.sum(xi, axis=1)
            a = torch.sum(a, axis=1)
            xi /= a[:, None, None]
            return xi

    def _log_likelihood(self, gamma, xi):
        likelihood = 0

        likelihood += torch.sum(gamma[0] * torch.log(self.initial_pi))

        for t in range(len(self.outputs)):
            for i in range(self.num_states):
                likelihood += gamma[t, i] * torch.log(self.dnorm(self.outputs[t], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[t]))), self.sd[i]))
                for j in range(self.num_states):
                    likelihood += xi[t, i, j] * torch.log(self.softmax(self.inputs[t], j))[i]

        return likelihood

    def _baum_welch(self):
        # sgd not good, better LBFGS
        optimizer = optim.LBFGS([self.initial_pi, self.transition_matrix, self.emission_matrix, self.sd], lr=1.0)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Decrease lr by 0.9 every 1 steps

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

            optimizer.step(closure)
            
            with torch.no_grad():
                new_log_likelihood = self._log_likelihood(gamma, xi)
                
                # Check for convergence
                if torch.abs(new_log_likelihood - old_log_likelihood) < self.tol:
                    print("convergence reached :)")
                    break
                old_log_likelihood = new_log_likelihood
            print(i + 1, new_log_likelihood.item())
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
                transition_prob = self.softmax(U[i + 1], last_state)
                last_state = torch.argmax(torch.log(transition_prob) + alpha[i])
                path.append(last_state.item())

            path.reverse()
            return path

    def predict(self, input):
        pass
