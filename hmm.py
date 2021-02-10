from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here

        for s_idx in range(S):
            alpha[s_idx,0] = self.pi[s_idx] * self.B[s_idx,self.obs_dict[Osequence[0]]]

        for t in range(1,L):
            for s_idx in range(S):
                prev_sum = 0

                for s_prev_idx in range(S):
                    prev_sum+= self.A[s_prev_idx,s_idx] * alpha[s_prev_idx,t-1]

                alpha[s_idx,t] = self.B[s_idx, self.obs_dict[Osequence[t]]] * prev_sum
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here

        for s_idx in range(S):
            beta[s_idx,L-1] = 1

        for t in range(L-2,-1,-1):
            for s_idx in range(S):
                sum=0
                for s_next_idx in range(S):
                    sum += self.A[s_idx,s_next_idx] * self.B[s_next_idx, self.obs_dict[Osequence[t+1]]] * beta[s_next_idx,t+1]

                beta[s_idx,t] = sum
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        prob = 0
        S,L = alpha.shape
        # for s_idx in range(S):
        #     prob+=alpha[s_idx,L-1]
        prob = sum(alpha[:,L-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        prob = self.forward(Osequence) * self.backward(Osequence) / self.sequence_prob(Osequence)
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here

        seq_prob = self.sequence_prob(Osequence)

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)

        for s in range(S):
            for s_dash in range(S):
                for t in range(0,L-1):
                    prob[s,s_dash,t] = (alpha[s,t] * self.B[s_dash,self.obs_dict[Osequence[t+1]]] * self.A[s,s_dash] * beta[s_dash,t+1] )/seq_prob

        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here

        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S,L])
        path_map = np.zeros([S,L],dtype='int')

        for s_idx in range(S):
            delta[s_idx,0] = self.pi[s_idx] * self.B[s_idx,self.obs_dict[Osequence[0]]]
            path_map[s_idx,0] = -1

        for t in range(1,L):

            for s_idx in range(S):
                temp=[]
                for s_prev_idx in range(S):
                    temp.append( self.B[s_idx,self.obs_dict[Osequence[t]]] * self.A[s_prev_idx,s_idx] * delta[s_prev_idx,t-1])
                delta[s_idx,t] = max(temp)
                path_map[s_idx,t] = np.argmax(temp)

        path_idx=[]
        path_idx.append(np.argmax(delta[:,L-1]))
        for t in range(L-1,0,-1):
            path_idx.append(path_map[path_idx[-1],t])

        key_list = list(self.state_dict.keys())
        vals_list = list(self.state_dict.values())
        for i in range(len(path_idx)-1,-1,-1):
            path.append(key_list[vals_list.index(path_idx[i])])

        ###################################################
        return path
