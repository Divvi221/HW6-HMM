import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        T = len(input_observation_states) #["sunny","rainy","rainy","sunny","rainy"]
        N = len(self.hidden_states) #hot and cold are hidden states
        #print(input_observation_states) 
        forward = np.zeros((N,T)) #initial forward prob matrix 
        
        #initialize forward probabilities:
        for s in range(N):
            obs_ind = np.where(self.observation_states == input_observation_states[0])[0][0] #find index of the first input obs states in the list of obs states 
            #print(obs_ind)
            forward[s,0] = self.prior_p[s] * self.emission_p[s,obs_ind]
        #print(forward)
        # Step 2. Calculate probabilities
        for t in range(1,T):
            obs_ind = np.where(self.observation_states == input_observation_states[t])[0][0] #index of the observed output in the hidden state list
            #print(obs_ind)
            for s in range(N):
                #forward[s,t] = 
                sum = 0
                for s_p in range(N):
                    elem = forward[s_p,t-1] * self.transition_p[s_p,s] * self.emission_p[s,obs_ind] #changed s_p in emission to s
                    #if s_p==0:
                        #print(self.emission_p[s_p,obs_ind])
                    sum += elem
                forward[s,t] = sum
        #print(forward)
        # Step 3. Return final probability 
        forward_prob = 0
        for s in range(N):
            forward_prob += forward[s,T-1]
        #print(forward_prob)
        return forward_prob
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables

        #print(decode_observation_states)
        #print(self.hidden_states)

        #store probabilities of hidden state at each step 
        ##viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states))) they gave me this
        T = len(decode_observation_states) #["sunny","rainy","rainy","sunny","rainy"]
        N = len(self.hidden_states) #hot and cold are hidden states
        viterbi = np.zeros((N,T))
        pointer = np.zeros((N,T))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))         
        #print(viterbi)

        for s in range(N):
            obs_ind = np.where(self.observation_states == decode_observation_states[0])[0][0] #find index of the first input obs states in the list of obs states 
            viterbi[s,0] = self.prior_p[s] * self.emission_p[s,obs_ind]
            pointer[s,0] = 0
        #print(viterbi, pointer)
       # Step 2. Calculate Probabilities
        for t in range(1,T):
            obs_ind = np.where(self.observation_states == decode_observation_states[t])[0][0]
            #backpointer = 0
            for s in range(N):
                max_elem = 0
                for s_p in range(N):
                    elem = viterbi[s_p,t-1] * self.transition_p[s_p,s] * self.emission_p[s,obs_ind]
                    if elem>max_elem:
                        max_elem = elem
                        #print(max_elem)
                        #backpointer = s_p
                viterbi[s,t] = max_elem
                #pointer[s,t] = backpointer
        index = []
        for t in viterbi.T:
            #print(t)
            elem = max(t)
            ind = np.where(t==elem)[0][0]
            index.append(ind)
        
        for t in range(0,T):
            for s in range(N):
                pointer[s,t] = index[t]
        
        print(pointer)
        #print(viterbi)

        # Step 3. Traceback 
        best_prob = []
        for s in range(N):
            best_prob.append(viterbi[s,T-1])
        best_path_prob = max(best_prob)
        best_path_pointer = np.argmax(best_prob)
        
        #print(best_prob)
        #print(best_path_pointer)
        hidden_state_seq = []
        for i in pointer[best_path_pointer,:]:
            #print(i)
            #print(self.hidden_states)
            hidden_elem = self.hidden_states[int(i)]
            hidden_state_seq.append(hidden_elem)

        # Step 4. Return best hidden state sequence 
        return hidden_state_seq
        

#testing
full_hmm=np.load('./data/full_weather_hmm.npz')
full_input=np.load('./data/full_weather_sequences.npz')

hmm = HiddenMarkovModel(full_hmm["observation_states"],full_hmm["hidden_states"],full_hmm["prior_p"],full_hmm["transition_p"],full_hmm["emission_p"])
hmm_forward = hmm.forward(full_input["observation_state_sequence"])
hmm_vit = hmm.viterbi(full_input["observation_state_sequence"])

print(full_input["best_hidden_state_sequence"])
print(hmm_vit)