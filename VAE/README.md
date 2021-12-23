# VAE
https://arxiv.org/abs/1312.6114

## traditional probabilistic model trainning method 
1) MLE  
  If set X is given like X = {x1, x2, ..., xn}, we can define P(X) which probability that X will be extracted is max.  
  We make P(X) by using neural network and, define that weight 'W', we can write it as P(X|W).  
  W is calculated by using 
  <p align="center"> <img src="./img1/equation.png" alt="MLE" width="20%" height="20%"/> </p> and 'W' makes logP(X) max
  
    
2) MAP  
