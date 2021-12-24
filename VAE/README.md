# VAE
1) Auto-Encoding Variational Bayes : https://arxiv.org/abs/1312.6114
2) beta-VAE: https://openreview.net/forum?id=Sy2fzU9gl

## 1. Terms
  **P(X)** : It is the distribution that explain what is X. For example we want to make dog image generator. P(X) should have high value when X = "dog image". But it has low value when X = "not dog image".  
  
  **P(X|theta)** : Theta is network weight, so P(X|theta) means deep network.  
  
  **p(Z)** : It is prior probability, but we don't know that, so we decide shape arbitrarily and approximate p(z) to q(z|x).
  
  **q(z|X)** : It is the core concept of VAE paper. We don't know what is data distribution, so we assume q(z|x) specific data distribution such as normal, uniform distribution and so on. q(z|x) is approximate probability distribution of p(z).  
  
  VAE's goal is to make data distribution by training prior probability p(z) and posterior probability p(x|z)
  for making output image which we want to focus such as target image. It seems same to MAP.
  
  
## 2. Traditional probabilistic model training method 


### 2-1) MLE  
  If set X is given like **X = {x1, x2, ..., xn}**, we can define P(X) which probabiltiy of sampling set X is max.  
  We make P(X) by using neural network and, define that weight as 'W', so we can write it as P(X|W).    

   'W' which makes logP(X) max is calculated by
  <p align="center"> <img src="./img1/equation.png" alt="MLE" width="20%" height="20%"/> </p> 

  Use stochastic gradient method for traning model, we can define loss function like that
  <p align="center"> <img src="./img1/equation (1).png" alt="MLE" width="40%" height="40%"/> </p> 

  So, 'W' can be updated by using loss function gradient and get converged 'W'  
  When on network's input set is X, and predicted output set is Y', we can define network model as P(Y = Y' | X, W)

  At first, i thought that if i use MLE, model wouldn't be trained because MLE does not use predicted value.   
  But that mathematical expression includes predicted value.

  * regression form
  <p align="center"> <img src="./img1/equation (3).png" alt="MLE" width="70%" height="70%"/> </p> 

  * classification form
  <p align="center"> <img src="./img1/equation (4).png" alt="MLE" width="60%" height="60%"/> </p> 
  
  
  
### 2-2) MAP
  MAP use bayesian rule
  <p align="center"> <img src="./img1/equation (5).png" alt="MLE" width="40%" height="40%"/>

  Its goal is to maximize P(w|X), so it is same to maximize P(X|w) just like MLE.  
  But it has difference with MLE that MAP consider probability distribution P(w) when maximize P(X|w)  
  MLE consider P(w) is uniform distribution, but MAP doesn't, so Those two method's results are different. 

  We can summarize MAP that is "MLE plus constraints" on P(W). Calculating w by using MAP is same to calculating w by using MLE "when w is regularized."
  <p align="center"> <img src="./img1/equation (6).png" alt="MLE" width="70%" height="70%"/> </p> 
  
  
  
## 3. What is differnece between VAE and MLE ?
  Why use variational inference for training in VAE? Why just use MLE for training that model?  
  For example, imagine making random latent vector z and matching it 'x'
  
  when generator model is 'G' target image set is X and weight is 'theta', p(x) is defined like that 
  <p align="center"> <img src="./img1/equation (7).png" alt="MLE" width="40%" height="40%"/> </p> 
  
  If we trained P(X) by using MLE method, we could train it by making pair (x_i, z_i) and calculated 'theta' for maximizing P(X).   But this method doesn't gurantee that network's output "predicted x" is target image's kind about new input z' not used for train. Because there is no relation between x and z.  
  
  For example, G(z_1) = x_1, G(z_2) = x_2, and G(z_3) = x_3. G is generator, and x_1, x_2, x_3 is predicted image. Assume distance function d, lower d(x,y) closer x and y. 
  **If d(x_1, x_2) > d(x_2, x_3) is correct, does it gurantee that d(z_1, z_2) > d(z_2, z_3)? Can we compare z_1, z_2 and z_3?**
  
  So, We use sampling function for giving relationship between z and x. 
  
  To implement VAE, paper suggest encoder and decoder method. So in this VAE code, i use encoder, decoder method to implement VAE.
  
### 3-1) Encoder
  Encoder is defined q(z|x) in VAE, and its fuction is sampling z from x for making relationship x and z. So we do not need to randomly match x and z for training. It solves the problem of training network by using MLE method.   
  
### 3-2) Decoder
 Decoder is to make target image by using latent code made by encoder. 
 

Encoder, decoder process is equal to prior probability and likelihood probability.
 
## 4. What is differnece between GAN and VAE ?
 
