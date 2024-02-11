# NEAT-JAX
An implementation of Neuroevolution of Augmented Topologies(NEAT) Algorithm compatible with EvoJAX. Made to play Neural Slime Volleyball 

## NEAT in JAX ? Why ?
Neuroevolution of Augmented Topologies(NEAT) is a biologically inspired method for evolving neural architectures. Instead of using backprop to update the weights of the network, NEAT uses Genetic Algorithms(GA) to evolve both neural network topology AND weights !! So cool.

NEAT, or GA in general are not really the most popular techniques that are in use in a field that is dominated by deep learning, but still a lot work has been done in this field. I did this as a passion project, since I wanted to have a NEAT solver that is compatible with EvoJAX.

## What is EvoJAX ?
EvoJAX is a scalable, general purpose, hardware-accelerated neuroevolution toolkit. Built on top of the JAX library, this toolkit enables neuroevolution algorithms to work with neural networks running in parallel across multiple TPU/GPUs. It offers a all-in-one library for deploying neuro-evolution techniques for wide variety of tasks, and is developed Yujin Tang, Yingtao Tian and David Ha.

![68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f6150697775466a7839664b654879494642482f67697068792e676966](https://github.com/yash-srivastava19/NEAT-JAX/assets/85068689/121d7c39-e4cf-4310-954b-cd8cd0ca8b18)


EvoJAX offers a really nice framework for neuroevolution algorithms, and making NEAT compatible with it open new door for testing, say for example RL enviroments such as Neural Slime Volleyball.

**UPDATE** : There seems to be some issue with `ind.py`/`neat.py` file where I think errors as raised because we are working with both Python Lists and JAX arrays. Any help in that regards would be really beneficial, and would go a long way in implementing the training notebook of NEAT agents(see the `neat_jax.ipynb`) 
