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
                    elem = forward[s_p,t-1] * self.transition_p[s_p,s] * self.emission_p[s_p,obs_ind]
                    #if s_p==0:
                        #print(self.emission_p[s_p,obs_ind])
                    sum += elem
                forward[s,t] = sum
        #print(forward)
        # Step 3. Return final probability 
        forward_prob = 0
        for s in range(N):
            forward_prob += forward[s,T-1]
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
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))         
        
       
       # Step 2. Calculate Probabilities

            
        # Step 3. Traceback 


        # Step 4. Return best hidden state sequence 
        
#testing and debugging
mini_hmm=np.load('./data/mini_weather_hmm.npz')
mini_hmm_seq=np.load('./data/mini_weather_sequences.npz')
#print(mini_hmm.keys())
hmm = HiddenMarkovModel(mini_hmm["observation_states"],mini_hmm["hidden_states"],mini_hmm["prior_p"],mini_hmm["transition_p"],mini_hmm["emission_p"])
hmm_forward = hmm.forward(mini_hmm_seq["observation_state_sequence"])