# VAE
https://arxiv.org/abs/1312.6114

## Traditional probabilistic model training method 
### 1. MLE  
  If set X is given like X = {x1, x2, ..., xn}, we can define P(X) which probabiltiy of sampling set X is max.
  We make P(X) by using neural network and, define that weight as 'W', so we can write it as P(X|W).    
  
   'W' which makes logP(X) max is calculated by
  <p align="center"> <img src="./img1/equation.png" alt="MLE" width="20%" height="20%"/> </p> 
  
  Use stochastic gradient method for traning model, we can define loss function like that
  <p align="center"> <img src="./img1/equation (1).png" alt="MLE" width="40%" height="40%"/> </p> 
  
  So, 'W' can be updated by using loss function gradient and get converged 'W'  
  When on network's input set is X, and predicted output set is Y', we can define network model as P(Y = Y' | X, W)
  
  At first, i thought that if i use MLE, model wouldn't be trained because MLE does not use predicted value 
  But, that mathematical expression includes predicted value.
  
  * regression form
  <p align="center"> <img src="./img1/equation (3).png" alt="MLE" width="60%" height="60%"/> </p> 
  
  * classification form
  <p align="center"> <img src="./img1/equation (4).png" alt="MLE" width="60%" height="60%"/> </p> 
 
### 2. MAP
  MAP use bayesian rule
  <p align="center"> <img src="./img1/equation (5).png" alt="MLE" width="40%" height="40%"/>
  
  Its goal is to maximize P(w|X), so it is same to maximize P(X|w) just like MLE.  
  But it has difference with MLE that MAP consider probability distribution P(w) when maximize P(X|w)  
  MLE consider P(w) is uniform distribution, but MAP doesn't, so Those two method's results are different. 
  
  We can summarize MAP that is "MLE plus constraints" on P(W). Calculating w by using MAP is same to calculating w by using MLE "when w is regularized."
  <p align="center"> <img src="./img1/equation (6).png" alt="MLE" width="60%" height="60%"/> </p> 
  
