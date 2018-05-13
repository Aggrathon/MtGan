# Magic The GAN
A neural network for generating playing cards.  
The project utilizes a GAN architechture for the NN.


## Dependencies
- Python 3
- Tensorflow
- Pandas


## Usage
1. Download and preprocess the data:   
`python prepare_data.py`   
(It will take hours to download ~33 000 images)
2. Train the model:   
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
- https://github.com/khanrc/tf.gans-comparison
- BEGAN: https://arxiv.org/abs/1703.10717
- WGAN-GP: https://arxiv.org/abs/1704.00028
- PG-GAN: https://arxiv.org/abs/1710.10196
