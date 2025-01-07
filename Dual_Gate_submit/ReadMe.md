This is the mini-project of 1677370, called Dual Gradient Gating: Alleviating Over-smoothing and Over-squashing in Graph Neural Networks

### Requirements
Main dependencies (with python >= 3.7)
torch==1.9.0
torch-cluster==1.5.9
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1

### Experiment
experiment include three part: `oversmooth`,`oversquash` and `heterophilic_graphs`, where both `oversmooth` and `heterophilic_graphs` are modified from github repository of paper `GRADIENT GATING FOR DEEP MULTI-RATE LEARNING ON GRAPHS` and `oversquash`are from paper `On Over-Squashing in Message Passing Neural Networks: The Impact of Width, Depth, and Topology`