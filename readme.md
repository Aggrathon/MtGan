# Magic The GAN
A neural network for generating playing cards.  
The project utilizes a GAN architechture for the NN.  
The specific GAN architechture is inspired by BEGAN (https://arxiv.org/abs/1703.10717).


## Dependencies
- Python 3
- Tensorflow
- Pandas


## Usage
1. Download and preprocess the data:   
`python prepare_data.py`   
(It will take hours to download ~33 000 images)
2. Train the model (coming soon):   
`python train.py`
3. Generate cards (coming later):   
`python generate.py`


## Links
These sources have provided inspiration and information during this project:
- https://www.gamasutra.com/blogs/AndyWallace/20171122/310166/Urzas_Dream_Engine__The_RoboRosewater_RoboDraft_Creating_a_machine_learning_algorithm_to_illustrate_Magic_cards.php
- http://mtgjson.com
- http://gatherer.wizards.com
- http://guimperarnau.com/blog/2017/11/Fantastic-GANs-and-where-to-find-them-II
- https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/
- https://arxiv.org/abs/1703.10717
