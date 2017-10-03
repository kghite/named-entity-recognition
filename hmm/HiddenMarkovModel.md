# Hidden Markov Model

## Model Overview

training sequence -> train transition probs and emission probs models
test sequence -> start probs, get emission probs, decode

## Decoder

*Input*

* Input Sequence: Observation set of test or plaintext data [o_1, o_2, ...]
	* Test Data: Tagged text gets converted to a list of word vectors to decode and a list of labels to check the accuracy
	* Plaintext Data: Text gets converted to a list of word vectors to decode
* Starting Probabilities: Dictionary of tags with inital probability of appearing for a given observation
	* ```
	  s_probs = { tag_1: 0.23, tag_2: 0.47, ... }
	  ```
* Transition Probabilities: Dictionary of tags with a dictionary of tag pairs mapped to the probability of moving to the key tag frome the pair
	* ```
	  t_probs = { tag_1: { 'tag_1 tag_2': 0.23, 
						   'tag_2 tag_3': 0.45,
						   ...					}
				  ...							}
	  ```
* Emission Probabilities: Dictionary of tags mapped to dictionary of observation wordvecs mapped to their emission probability for that tag
	* ```
	  e_probs = { tag_1: { o_1: 0.23, o_2: 0.47, ... }
				  ...								 }
	  ``` 

*Output*

* Decoded Tags: List of tags corresponding to the input text sequence
