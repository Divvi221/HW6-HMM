import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm = HiddenMarkovModel(mini_hmm["observation_states"],mini_hmm["hidden_states"],mini_hmm["prior_p"],mini_hmm["transition_p"],mini_hmm["emission_p"])
    hmm_forward = hmm.forward(mini_input["observation_state_sequence"])
    hmm_vit = hmm.viterbi(mini_input["observation_state_sequence"])
    #print(mini_input["best_hidden_state_sequence"])
    tol = 1e-4
    hmm_forward_true = 0.03506 #hand calculated probability
    assert (hmm_forward - 0.03506) <= tol #checking forward prob
    assert np.all(hmm_vit == mini_input["best_hidden_state_sequence"]) #checking to see if the hidden sequence is correct
    assert len(hmm_vit) == len(mini_input["best_hidden_state_sequence"]) #checking length of hidden sequence



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    pass


test_mini_weather()









