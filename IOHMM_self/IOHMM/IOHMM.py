import numpy as np, scipy as sp
from scipy.special import logsumexp
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

torch.autograd.set_detect_anomaly(True)

# assuming a linear regression emission model and a logistic regression transition model

class IOHMM_model:
    def __init__(self, num_states, inputs, outputs, max_iter, tol, initial_pi=None, transition_matrix=None, emission_matrix=None, lsd=None):
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
            self.initial_pi = nn.Parameter(torch.ones(num_states) / num_states, requires_grad=True)
            # self.initial_pi = torch.ones(num_states) / num_states
        if transition_matrix is not None:
            self.transition_matrix = nn.Parameter(transition_matrix, requires_grad=True)
        else:
            self.transition_matrix = nn.Parameter(torch.randn(num_states, num_states, inputs.shape[1] + 1), requires_grad=True)
        if emission_matrix is not None:
            self.emission_matrix = nn.Parameter(emission_matrix, requires_grad=True)
        else:
            self.emission_matrix = nn.Parameter(torch.randn(num_states, inputs.shape[1] + 1), requires_grad=True)
        if lsd is not None:
            self.lsd = nn.Parameter(lsd, requires_grad=True)
        else:
            # setting a higher lsd was fundamental to make the model stable, otherwise the emission prob would be zero
            self.lsd = nn.Parameter(5.0*torch.ones(num_states), requires_grad=True)


    def softmax_(self, input, state):
        with torch.no_grad():
            return torch.exp(self.transition_matrix[state,:,:] @ torch.cat((torch.tensor([1.0]),input))) / torch.sum(torch.exp(self.transition_matrix[state,:,:] @ torch.cat((torch.tensor([1.0]),input))))
                
    def softmax(self, input):
        with torch.no_grad():
        
            # Concatenate 1 to the beginning of the input
            input_with_bias = torch.cat((torch.tensor([1.0]), input))

            # Compute transition probabilities for all states
            transition_logits = self.transition_matrix @ input_with_bias

            # Compute softmax probabilities
            exp_logits = torch.exp(transition_logits)
            softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

            return softmax_probs

    def dnorm(self, x, mean, lsd):
        with torch.no_grad():
            return torch.exp(-0.5 * ((x - mean) / torch.exp(lsd)) ** 2) / (torch.exp(lsd) * torch.sqrt(torch.tensor(2 * np.pi)))

    def _forward(self):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            alpha = torch.zeros((T, N))

            # Initialize base cases (t == 0)
            emission_prob = self.dnorm(self.outputs[0], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),U[0]))), self.lsd)
            # to normalize
            alpha[0] = self.initial_pi * emission_prob
            alpha[0] /= torch.sum(alpha[0])

            # Compute forward probabilities (t > 0)
            for t in range(1, T):
                emission_prob = self.dnorm(self.outputs[t], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),U[t]))), self.lsd)
                transition_prob = self.softmax(U[t])
                # sum on the columns of transition_prob, quite sure about axis = 0
                transition_prob *= alpha[t-1].unsqueeze(1)
                transition_prob = torch.sum(transition_prob, axis=0)
                alpha[t] = emission_prob * transition_prob
                # normalize
                alpha[t] /= torch.sum(alpha[t])

            return alpha


    def _backward(self):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            beta = torch.zeros((T, N))

            # Initialize base cases (t == T-1)
            # parallel version
            emission_prob = self.dnorm(self.outputs[-1], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),U[-1]))), self.lsd)
            # to normalize
            beta[-1] = emission_prob
            beta[-1] /= torch.sum(beta[-1])

            # Compute forward probabilities (t > 0)
            for t in reversed(range(T-1)):
                emission_prob = self.dnorm(self.outputs[t], self.emission_matrix @ (torch.cat((torch.tensor([1.0]),U[t]))), self.lsd)
                transition_prob = self.softmax(U[t])
                transition_prob *= beta[t+1].unsqueeze(1)
                # sum on the columns of transition_prob, quite sure about axis = 0
                transition_prob = torch.sum(transition_prob, axis=0)
                beta[t] = emission_prob * transition_prob
                # normlize
                beta[t] /= torch.sum(beta[t])

            return beta


    def _compute_gamma(self, alpha=None, beta=None):
        with torch.no_grad():
            # in the paper it says that zeta_it = P(x_t=i | all obs of inputs from 0 to t)
            # so it should simply be alpha, but not sure
            gamma = alpha * beta
            # normalize
            gamma /= torch.sum(gamma, axis=1).reshape(-1, 1)
            return gamma


    def _compute_xi(self, alpha=None, beta=None):
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            xi = torch.zeros((T, N, N))

            for t in range(0, T):
                transition_prob = self.softmax(U[t]).T
                # transpose to maintain the same notation of the article, in the row next state, in the columns previous state
                #print(f"transition prob: {transition_prob}")
                #print(f"alpha[t-1]: {alpha[t-1]}")
                #print(f"beta[t]: {beta[t]}")
                # not sure if i have to unsqueeze alpha or beta, is it the same?
                xi[t, :, :] = transition_prob * beta[t].unsqueeze(1) * alpha[t-1]

            # S = torch.sum(alpha[T-1, :], axis=0)
            # xi /= S

            # normalize
            # to check xi
            a = torch.sum(xi, axis=1)
            a = torch.sum(a, axis=1)
            xi /= a[:, None, None]
            return xi
    
    def _log_likelihood(self, gamma, xi):
        likelihood = 0
        likelihood += torch.sum(gamma[0] * torch.log(self.initial_pi))
        for t, output in enumerate(self.outputs):
            # first term of the sum

            x = self.emission_matrix @ (torch.cat((torch.tensor([1.0]), self.inputs[t])))
            mu = output


            dnorm = -0.5 * (((mu - x) / torch.exp(self.lsd)) ** 2) / (torch.exp(self.lsd) * torch.sqrt(torch.tensor(2 * np.pi)))

            likelihood += torch.sum(gamma[t, :] * dnorm)
            # second term of the sum

            # Concatenate 1 to the beginning of the input
            input_with_bias = torch.cat((torch.tensor([1.0]), self.inputs[t]))
            # Compute transition probabilities for all states
            transition_logits = self.transition_matrix @ input_with_bias
            # Compute softmax probabilities
            exp_logits = torch.exp(transition_logits)
            softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
            # to check relation between xi and softmax
            likelihood += torch.sum(xi[t, :, :] * torch.log(softmax_probs))
                
        return likelihood



    def _baum_welch(self):
        # sgd not good, better LBFGS
        optimizer = optim.SGD([self.initial_pi, self.transition_matrix, self.emission_matrix, self.lsd], lr=0.1)
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
                transition_prob = self.softmax(U[i + 1], last_state)
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
        
        
"""
    def _log_likelihood(self, gamma, xi):
        likelihood = 0

        # Initial state log likelihood
        likelihood += torch.sum(gamma[0] * torch.log(self.initial_pi))

        # missing a sum over p, that are different sequnces of hmm, but we are assuming only one sequence
        # Emission log likelihood
        # ones = torch.ones((self.inputs.size(0), 1))

        # Concatenate ones with self.inputs along the columns
        # inputs_bias = torch.cat((ones, self.inputs), dim=1)  # Shape (T, N+1)

        # Expand dimensions to ensure compatibility
        # outputs_expanded = self.outputs[:, None]  # Shape (T, 1)
        # inputs_bias_expanded = inputs_bias[:, None, :]  # Shape (T, 1, N+1)

        # Compute the mean for dnorm
        # mean = self.emission_matrix @ inputs_bias_expanded.transpose(1, 2)  # Shape (T, M, 1)

        # Print log of dnorm
        # print(torch.log(self.dnorm(outputs_expanded, mean.squeeze(-1), self.sd)))

        # Update likelihood
        # likelihood += torch.sum(gamma * torch.log(self.dnorm(outputs_expanded, mean.squeeze(-1), self.sd)))
        
        # likelihood += torch.sum(xi * torch.log(self.softmax(self.inputs)))

        for t in range(len(self.outputs)):
            print(torch.sum(gamma[t, :] * torch.log(self.dnorm(self.outputs[t], self.emission_matrix*(torch.cat((torch.tensor([1.0]), self.inputs[t])))[:,None], self.sd))))
            print(torch.sum(xi[t, :, :] * torch.log(self.softmax(self.inputs[t]))))

            likelihood += torch.sum(gamma[t, :] * torch.log(self.dnorm(self.outputs[t], self.emission_matrix*(torch.cat((torch.tensor([1.0]), self.inputs[t])))[:,None], self.sd)))

            likelihood += torch.sum(xi[t, :, :] * torch.log(self.softmax(self.inputs[t])))

        return likelihood


    


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