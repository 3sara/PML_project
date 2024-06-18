import numpy as np, scipy as sp
from scipy.special import logsumexp
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

torch.autograd.set_detect_anomaly(True)

class IOHMM_model:
    def __init__(self, num_states, inputs, outputs, max_iter, tol, log_initial_pi=None, transition_matrix=None, emission_matrix=None, log_sd=None):
        self.num_states = num_states
        self.inputs = inputs
        self.outputs = outputs
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

        """
            Initial state distribution: size (num_states)
                the initial state distribution, if None, it is initialized as a uniform distribution

            Transition matrix: size (num_states, num_states, inputs.shape[1]+1)
                the coefficients for a logistic regression model, if None, it is initialized randomly
                first dimension is the current state, second dimension is the next state, third dimension is the input
                the dimension of the input is inputs.shape[1] + 1, since we include
            
            Emission matrix:
                the coefficients for a linear regression model, size is (num_states, outputs.shape[0])
        """


        if log_initial_pi is not None:
            self.log_initial_pi = nn.Parameter(log_initial_pi, requires_grad=True)
        else:
            self.log_initial_pi = nn.Parameter(torch.log(torch.ones(num_states)/num_states), requires_grad=True)

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

    def log_normalization(self, x):
        return x - torch.logsumexp(x, dim=0)


    def log_softmax_(self, input, state):
        """
            Compute the log softmax of the transition matrix for a given input and state.
        """
        with torch.no_grad():

            return (self.transition_matrix[state,:,:] @ torch.cat((torch.tensor([1.0]),input))) - torch.logsumexp(torch.exp(self.transition_matrix[state,:,:] @ torch.cat((torch.tensor([1.0]),input))), dim =0)
    
                
    def log_softmax(self, input):
        """
            Compute the log softmax of the transition matrix for a given input.
            Input:
                input: the input, size is (input.shape[0])
        """
        with torch.no_grad():
        
            # Concatenate 1 to the beginning of the input
            input_with_bias = torch.cat((torch.tensor([1.0]), input))

            transition_logits = self.transition_matrix @ input_with_bias

            log_softmax_probs = transition_logits - torch.logsumexp(transition_logits, dim=1)

            return log_softmax_probs

    
    def log_dnorm(self, x, mean, log_sd):
        """
            Compute the log density of a normal distribution with mean and log standard deviation.
        """
        with torch.no_grad():

            return (-0.5 * ((x - mean) / torch.exp(log_sd)) ** 2) - (log_sd + torch.log(torch.sqrt(torch.tensor(2 * np.pi))))

    
    def _forward(self):
        """
            Compute the forward probabilities.
            Returns:
                log_alpha: the log forward probabilities, size is (T, num_states)
        """

        with torch.no_grad():
            
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            log_alpha = torch.zeros((T, N))

            # Initialization (t == 0)
            input_with_bias = torch.cat((torch.tensor([1.0]), U[0]))
            log_emission_prob = self.log_dnorm(self.outputs[0], self.emission_matrix @ input_with_bias, self.log_sd)
            log_alpha[0] = self.log_normalization(self.log_initial_pi) + log_emission_prob
            # Normalization
            log_alpha[0] -= torch.logsumexp(log_alpha[0], dim = 0)

            # Iteration (t > 0)
            for t in range(1, T):
                input_with_bias = torch.cat((torch.tensor([1.0]), U[t]))
                log_emission_prob = self.log_dnorm(self.outputs[t], self.emission_matrix @ input_with_bias, self.log_sd)
                log_transition_prob = self.log_softmax(U[t])
                log_transition_prob += log_alpha[t-1].unsqueeze(1)
                log_alpha[t] = log_emission_prob + torch.logsumexp(log_transition_prob, axis=0)
                # Normalization
                log_alpha[t] -= torch.logsumexp(log_alpha[t], dim=0)
            
            return log_alpha


    def _backward(self):
        """
            Compute the backward probabilities.
            Returns:
                log_beta: the log backward probabilities, size is (T, num_states)
        """
        
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            log_beta = torch.zeros((T, N))

            # Initialization (t == T-1)
            input_with_bias = torch.cat((torch.tensor([1.0]), U[-1]))
            log_emission_prob = self.log_dnorm(self.outputs[-1], self.emission_matrix @ input_with_bias, self.log_sd)
            log_beta[-1] = log_emission_prob
            # Normalization
            log_beta[-1] -= torch.logsumexp(log_beta[-1],dim=0)

            # Iteration (t > 0)
            for t in reversed(range(T-1)):
                input_with_bias = torch.cat((torch.tensor([1.0]), U[t]))
                log_emission_prob = self.log_dnorm(self.outputs[t], self.emission_matrix @ input_with_bias, self.log_sd)
                log_transition_prob = self.log_softmax(U[t])
                log_transition_prob += log_beta[t+1].unsqueeze(1)
                log_beta[t] = log_emission_prob + torch.logsumexp(log_transition_prob, axis=0)
                # Normalization
                log_beta[t] -= torch.logsumexp(log_beta[t],dim=0)

            return log_beta

    
    def _compute_log_gamma(self, log_alpha=None, log_beta=None):
        """
            Compute the gamma probabilities which are defined as:
                gamma_t(i) = P(X_t = i | Y, U)
            Returns:
                log_gamma: the log gamma probabilities, size is (T, num_states)
        """
        with torch.no_grad():

            log_gamma = log_alpha + log_beta
            # Normalization
            log_gamma -= torch.logsumexp(log_gamma, axis=1).reshape(-1, 1)

            return log_gamma
    

    def _compute_log_zeta(self):
        """
            Compute the zeta probabilities which are defined as:
                zeta_t(i) = P(X_t = i | Y_{1:t})
            Returns:
                log_zeta: the log zeta probabilities, size is (T, num_states)
        """
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            log_zeta = torch.zeros((T, N))

            log_zeta[0] = self.log_normalization(self.log_initial_pi) #approximation
            print(f"zeta[0]: {torch.exp(log_zeta[0])}")

            for t in range(1,T):
                log_zeta[t] = torch.logsumexp(log_zeta[t-1] + self.log_softmax(U[t]), axis=1)
                # Normalization
                log_zeta[t] -= torch.logsumexp(log_zeta[t], dim=0)

            return log_zeta


    def _compute_log_xi(self, log_alpha=None, log_beta=None):
        """
            Compute the xi probabilities which are defined as:
                xi_t(i,j) = P(X_t = i, X_t+1 = j | Y, U)
            Returns:
                log_xi: the log xi probabilities, size is (T, num_states, num_states)
        """
        
        with torch.no_grad():
            T = len(self.outputs)
            N = self.num_states
            U = self.inputs

            log_xi = torch.zeros((T, N, N))

            for t in range(0, T):
                transition_prob = self.log_softmax(U[t]).T
                log_xi[t, :, :] = transition_prob + log_beta[t].unsqueeze(1) + log_alpha[t-1]

            # Normalization
            a = torch.logsumexp(log_xi, axis=1)
            log_xi -= a[:, None]

            return log_xi

    def _log_likelihood(self, log_zeta, log_xi):
        
        l = 0

        zeta = torch.exp(log_zeta)
        xi = torch.exp(log_xi)

        log_dnorm_prob = torch.zeros((self.num_states))
        for i in  range(self.num_states):
            log_dnorm_prob[i] = self.log_dnorm(self.outputs[0],
                                               self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[0]))),
                                               self.log_sd[i])

        l += torch.sum(zeta[0] * log_dnorm_prob) + torch.sum(xi[0] * self.log_normalization(self.log_initial_pi))

        for t in range(1,len(self.outputs)):
            for i in  range(self.num_states):
                
                log_dnorm_prob = self.log_dnorm(self.outputs[t],
                                                self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[t]))),
                                                self.log_sd[i])

                l += zeta[t,i] * log_dnorm_prob

                for j in range(self.num_states):
                    l += xi[t,i,j] * self.log_softmax(self.inputs[t])[i,j]

        return l
    
    def _log_likelihood_(self, gamma, xi):
        likelihood = 0
        likelihood += torch.sum(torch.exp(gamma[0]) * self.log_normalization(self.log_initial_pi))
        
        for t, output in enumerate(self.outputs):
            
            # first term of the sum

            x = self.emission_matrix @ (torch.cat((torch.tensor([1.0]), self.inputs[t])))
            mu = output

            dnorm = (-0.5 * (((mu - x) / torch.exp(self.log_sd)) ** 2)) - ((self.log_sd) + torch.log(torch.sqrt(torch.tensor(2 * np.pi))))

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
        optimizer = optim.SGD([self.log_initial_pi, self.transition_matrix, self.emission_matrix, self.log_sd], lr=0.01)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)  # Decrease lr by 0.9 every 1 steps
        
        old_log_likelihood = -torch.inf

        for i in range(self.max_iter):

            # E-step: Compute the posterior probabilities
            log_alpha = self._forward()
            log_beta = self._backward()
            log_gamma = self._compute_log_gamma(log_alpha, log_beta)
            log_xi = self._compute_log_xi(log_alpha, log_beta)
            log_zeta = self._compute_log_zeta()

            # M-step
            def closure():
                optimizer.zero_grad()
                loss = -self._log_likelihood(log_zeta, log_xi)
                loss.backward()
                return loss

            optimizer.step(closure)
            #scheduler.step()
            
            with torch.no_grad():
                new_log_likelihood = self._log_likelihood(log_zeta, log_xi)
                l_marco = self._log_likelihood_(torch.exp(log_gamma), torch.exp(log_xi))
                print(f"iter: {i}, likelihood: {new_log_likelihood}, likelihood marco: {l_marco}")
                self.history.append(new_log_likelihood)
                # Check for convergence
                if torch.abs(new_log_likelihood - old_log_likelihood) < self.tol:
                    print("convergence reached :)")
                    print(f"final likelihood: {new_log_likelihood}")
                    break
                old_log_likelihood = new_log_likelihood
            

        if i == self.max_iter:    
            print("convergence not reached")

    def _viterbi(self):
        with torch.no_grad():
            U = self.inputs

            log_alpha = self._forward()
            path = []
            last_state = torch.argmax(log_alpha[-1])
            path.append(last_state.item())

            for i in reversed(range(len(self.outputs) - 1)):
                transition_prob = self.log_softmax_(U[i + 1], last_state)
                last_state = torch.argmax(torch.log(transition_prob) + log_alpha[i])
                path.append(last_state.item())

            path.reverse()
            return path
    
    def viterbi(self):

            prob=torch.zeros((len(self.outputs),self.num_states), dtype=torch.float64)
            path=torch.zeros((len(self.outputs),self.num_states), dtype=torch.int64)
            
            #initialize the first state
            for i in range(self.num_states):
                prob[0,i] = self.log_normalization(self.log_initial_pi)[i] + self.log_dnorm(self.outputs[0], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[0]))), self.log_sd[i])

            #complete the matrixes
            for t in range(1,len(self.outputs)):
                for i in range(self.num_states):
                    prob[t,i] = torch.max(prob[t-1,:] + self.log_softmax(self.inputs[t])[i] + self.log_dnorm(self.outputs[t], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[t]))), self.log_sd[i]))
                    path[t,i]=torch.argmax(prob[t-1,:] + self.log_softmax(self.inputs[t])[i] + self.log_dnorm(self.outputs[t], self.emission_matrix[i].dot(torch.cat((torch.tensor([1.0]), self.inputs[t]))), self.log_sd[i]))
            
            state_sequence=torch.zeros(len(self.outputs), dtype=torch.int64)
            state_sequence[-1]=torch.argmax(prob[-1,:])
            for t in range(len(self.outputs)-2,-1,-1):
                state_sequence[t]=path[t+1,state_sequence[t+1]]
            return state_sequence
        

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