import numpy as np, scipy as sp
from scipy.special import logsumexp
import torch
from torch import nn
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

# assuming a linear regression emission model and a logistic regression transition model

class IOHMM_model:
    def __init__(self, num_states, inputs, outputs, max_iter, tol):
        self.num_states = num_states
        self.inputs = inputs
        self.outputs = outputs
        self.max_iter = max_iter
        self.tol = tol

        # self.initial_pi = torch.ones(num_states) / num_states
        # transition matrix: the coefficients for a logistic regression model, size is (num_states, num_states, inputs.shape[0])
        # first dimension is the current state, second dimension is the next state, third dimension is the input
        # self.transition_matrix = 2 * torch.rand(num_states, num_states, inputs.shape[1], dtype=torch.float32) -1
        # emission matrix: the coefficients for a linear regression model, size is (num_states, outputs.shape[0])
        # self.emission_matrix = 2 * torch.rand(num_states, inputs.shape[1], dtype=torch.float32) -1

        self.initial_pi = nn.Parameter(torch.ones(num_states)/num_states, requires_grad=True)
        self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states, inputs.shape[1]+1), requires_grad=True)
        self.emission_matrix = nn.Parameter(torch.randn(num_states, inputs.shape[1]+1), requires_grad=True)
        self.sd = nn.Parameter(torch.ones(num_states), requires_grad=True)


    def softmax_(self, input, state):
        with torch.no_grad():
            return torch.exp(self.transition_matrix[state,:,:] @ torch.cat((torch.tensor([1.0]),input))) / torch.sum(torch.exp(self.transition_matrix[state,:,:] @ torch.cat((torch.tensor([1.0]),input))))
                
    def softmax(self, input):
        # Concatenate 1 to the beginning of the input
        input_with_bias = torch.cat((torch.tensor([1.0]), input))

        # Compute transition probabilities for all states
        transition_logits = self.transition_matrix @ input_with_bias

        # Compute softmax probabilities
        exp_logits = torch.exp(transition_logits)
        softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

        return softmax_probs

    def dnorm(self, x, mean, sd):
        with torch.no_grad():
            return torch.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * torch.sqrt(torch.tensor(2 * np.pi)))

    def _forward(self):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            alpha = torch.zeros((T, N))

            # Initialize base cases (t == 0)
            # i think this can be parallelized
            # for i in range(N):
            #    emission_prob = self.dnorm(self.outputs[0], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]),U[0]))), self.sd[i])
            #    alpha[0, i] = self.initial_pi[i] * emission_prob
            
            # parallel version
            emission_prob = self.dnorm(self.outputs[0], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),U[0]))), self.sd)
            # to normalize
            alpha[0] = self.initial_pi * emission_prob
            alpha[0] /= torch.sum(alpha[0])

            # Compute forward probabilities (t > 0)
            for t in range(1, T):
                # i think this can be parallelized
                emission_prob = self.dnorm(self.outputs[t], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),U[t]))), self.sd)
                transition_prob = self.softmax(U[t-1])
                # sum on the columns of transition_prob, quite sure about axis = 0
                transition_prob = torch.sum(transition_prob, axis=0)
                alpha[t] = emission_prob * transition_prob
                # maybe better no to normalize for performance                
                # alpha[t] /= torch.sum(alpha[t])

            return alpha


    def _backward(self):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            beta = torch.zeros((T, N))

            # Initialize base cases (t == T-1)
            # parallel version
            emission_prob = self.dnorm(self.outputs[-1], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),U[0]))), self.sd)
            # to normalize
            beta[-1] = emission_prob
            beta[-1] /= torch.sum(beta[-1])

                        # Compute forward probabilities (t > 0)
            for t in reversed(range(T-1)):
                # i think this can be parallelized
                emission_prob = self.dnorm(self.outputs[t+1], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),U[t+1]))), self.sd)
                transition_prob = self.softmax(U[t])
                # sum on the columns of transition_prob, quite sure about axis = 0
                transition_prob = torch.sum(transition_prob, axis=0)
                beta[t] = emission_prob * transition_prob
                # maybe better no to normalize for performance                
                # beta[t] /= torch.sum(beta[t])

            return beta


    def _compute_gamma(self, alpha=None, beta=None):
        with torch.no_grad():
            # in the paper it says that zeta_it = P(x_t=i | all obs of inputs from 0 to t)
            # so it should simply be alpha, but not sure
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
                    #for j in range(N):
                    #    transition_prob = self.softmax(U[t], j)
                    #    xi[t, i, j] = (beta[t, i] * alpha[t-1, j] * transition_prob[i]) / torch.sum(alpha[T-1])
                    transition_prob = self.softmax(U[t])
                    transition_prob = torch.sum(transition_prob, axis=0)
                    xi[t, i, :] = beta[t] * alpha[t-1] * transition_prob / torch.sum(alpha[T-1])


            # normalize
            a = torch.sum(xi, axis=1) 
            a = torch.sum(a, axis=1)
            a = a[:, None, None]
            xi /= a
            return xi


    def _log_likelihood(self, gamma, xi):
        likelihood = 0

        # Initial state log likelihood
        likelihood += torch.sum(gamma[0] * torch.log(self.initial_pi))

        # Emission log likelihood
        for t in range(len(self.outputs)):
            for i in range(self.num_states):
                likelihood += gamma[t, i] * torch.log(self.dnorm(self.outputs[t], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[t]))), self.sd[i]))
                for j in range(self.num_states):
                    likelihood += xi[t, i, j] * torch.log(self.softmax_(self.inputs[t], j))[i]
                # transition_prob = self.softmax(self.inputs[t])
                # transition_prob = torch.sum(transition_prob, axis=0)
                # likelihood += torch.sum(xi[t, i, :] * torch.log(self.softmax(self.inputs[t]))[i])


        # missing a sum over p, that are different sequnces of hmm, but we are assuming only one sequence
        return likelihood

    def _baum_welch(self):
        optimizer = optim.LBFGS([self.initial_pi, self.transition_matrix, self.emission_matrix, self.sd], lr=0.01)
        gamma = torch.zeros((len(self.outputs), self.num_states))
        xi = torch.zeros((len(self.outputs), self.num_states, self.num_states))
        
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
                    break
                old_log_likelihood = new_log_likelihood
        

    

    '''
    def viterbi(self):

        prob=np.zeros((len(self.outputs),self.num_states))
        path=np.zeros((len(self.outputs),self.num_states))
        #initialize the first state
        for i in range(self.num_states):
            prob[0,i]=self.initial_pi[i]*self.dnorm(self.outputs[0], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[0]))), self.sd[i])

        # parallel version... magari
        #prob[0] = self.dnorm(self.outputs[0], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),self.inputs[0])), self.sd))
        
        #complete the matrixes
        for t in range(1,len(self.outputs)):
            for i in range(self.num_states):
                prob[t,i]=np.max(prob[t-1,:]*self.softmax(self.inputs[t])[i]*self.dnorm(self.outputs[t], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[t]))), self.sd[i]))
                path[t,i]=np.argmax(prob[t-1,:]*self.softmax(self.inputs[t])[i]*self.dnorm(self.outputs[t], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[t]))), self.sd[i]))
        
        state_sequence=np.zeros(len(self.outputs))
        state_sequence[-1]=np.argmax(prob[-1,:])
        for t in range(len(self.outputs)-2,-1,-1):
            state_sequence[t]=path[t+1,state_sequence[t+1]]
        return state_sequence
    '''
    def viterbi(self):
        
        with torch.no_grad():
            prob = torch.zeros((len(self.outputs),self.num_states))
            path = torch.zeros((len(self.outputs), self.num_states), dtype=torch.long)  # Use long for indexing
            
            emission_prob = self.dnorm(self.outputs[0], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),self.inputs[0]))), self.sd)
            # to normalize
            prob[0] = self.initial_pi * emission_prob
            # Complete the matrices
            for t in range(1, len(self.outputs)):
                    transition_prob = self.softmax(self.inputs[t])  # Ensure softmax is applied correctly
                    transition_prob = torch.sum(transition_prob, axis=0)
                    prob[t] = torch.max(prob[t - 1, :] * transition_prob * self.dnorm(self.outputs[t], self.emission_matrix @ (torch.cat((torch.tensor([1.0]), self.inputs[t]))), self.sd))
                    path[t] = torch.argmax(prob[t - 1, :] * transition_prob * self.dnorm(self.outputs[t], self.emission_matrix@ (torch.cat((torch.tensor([1.0]), self.inputs[t]))), self.sd))
            
            state_sequence = torch.zeros(len(self.outputs), dtype=torch.long)  # Use long for indexing
            state_sequence[-1] = torch.argmax(prob[-1, :])
            for t in range(len(self.outputs) - 2, -1, -1):
                state_sequence[t] = path[t + 1, state_sequence[t + 1]]
        
        return state_sequence

    def predict(self, future_input):
        # Future output is the expected output given the future input
        future_output = torch.zeros(future_input.shape[0])

        for input in future_input:
            transition_prob = self.softmax(input)
            transition_prob = torch.sum(transition_prob, axis=0)
            state = torch.argmax(transition_prob)
            future_output += self.emission_matrix[state].dot(torch.cat((torch.tensor([1.0]), input)))
                

        return future_output


"""
    def _baum_welch(self):
        theta_old = [self.initial_pi, self.transition_matrix, self.emission_matrix]
        for i in range(self.max_iter):
            # E-step: Compute the posterior probabilities
            alpha = self._forward()
            beta = self._backward()
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(alpha, beta)

            # M-step: Optimize theta to maximize the log likelihood
            # Initialize new theta parameters
            new_pi = gamma[0]
            new_transition_matrix = torch.zeros_like(self.transition_matrix)
            new_emission_matrix = torch.zeros_like(self.emission_matrix)

            # Update transition probabilities
            for i in range(self.num_states):
                for j in range(self.num_states):
                    numerator = torch.sum(xi[:, i, j])
                    denominator = torch.sum(gamma[:-1, i])
                    new_transition_matrix[i, j] = numerator / denominator

            # Update emission probabilities
            for i in range(self.num_states):
                for j in range(len(self.outputs)):
                    new_emission_matrix[i, j] = torch.sum(gamma[:, i] * self.inputs[:, j]) / torch.sum(gamma[:, i])

            # Set the updated parameters
            self.initial_pi = new_pi
            self.transition_matrix = new_transition_matrix
            self.emission_matrix = new_emission_matrix

            # Check for convergence
            theta_new = [new_pi, new_transition_matrix, new_emission_matrix]
            if torch.abs(self._log_likelihood(gamma, xi, theta_new) - self._log_likelihood(gamma, xi, theta_old)) < self.tol:
                break
            else:
                theta_old = theta_new

        return theta_new

"""





