# VAE
https://arxiv.org/abs/1312.6114

## Traditional probabilistic model training method 
### 1. MLE  
  If set X is given like X = {x1, x2, ..., xn}, we can define P(X) which probabiltiy of sampling set X is max.
  We make P(X) by using neural network and, define that weight as 'W', so we can write it as P(X|W).    
  
   'W' which makes logP(X) max is calculated by
  <p align="center"> <img src="./img1/equation.png" alt="MLE" width="20%" height="20%"/> </p> 
  
  Use stochastic gradient method for traning model, we can define loss function like that
  <p align="center"> <img src="./img1/equation(1).png" alt="MLE" width="20%" height="20%"/> </p> 
  
  So, w can be updated by using loss function gradient and get converged 'W'  
  
  When on network's input set is X, and output set is Y, we can define network model as
  <p align="center"> <img src="./img1/equation(2).png" alt="MLE" width="20%" height="20%"/> </p> 
  
  At first, i thought that if i use MLE, model wouldn't be trained because MLE does not use predicted value 
  But, that mathematical expression includes predicted value.
  
  * regression form
  <p align="center"> <img src="./img1/equation(3).png" alt="MLE" width="20%" height="20%"/> </p> 
  
  * classification form
  <p align="center"> <img src="./img1/equation(4).png" alt="MLE" width="100%" height="100%"/> </p> 
    
### 2. MAP
  
