Discrete Flow Matching on 2D Checkerboard

Project Overview

This repository implements a Discrete Flow Matching model designed to learn and generate a 2D Checkerboard distribution. The project is a part of TU Berlin WS 25/26 PML project.
This submission corresponds to Mileszone 1 of the project.

There are 3 standalone scripts that can be run. "run_flow_personal.py" trains the model and makes a nice gif showcasing the points changing to our target distribution from noise. It also generates a visualization of marginal velocity field at some time t. The scripts "compare_activations.py" and "compare_valid_acc.py" use different activation functions and evaluate their impact on our trained model.


Sources

The following sources were used as inspiration:
https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_discrete_flow_matching.ipynb
https://drscotthawley.github.io/blog/posts/FlowModels.html
