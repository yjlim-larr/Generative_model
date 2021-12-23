# Marginal distribution 
Joint probability distribution P(X, Z)'s marginal distribution is P(X) and P(Z)  
<p align="center"> <img src="./img1/equation1.png" alt="MLE" width="20%" height="20%"/> </p> 

# Conditional probability
Conditional probability P(X|Z) satisfies the following relationship.
<p align="center"> <img src="./img1/equation2.png" alt="MLE" width="20%" height="20%"/> </p> 

we use that relationship for define VAE loss function.

# Derive loss function
1. There are two random variable. 1) X, 2) Z. 
2. X and Z has a relationship.  

<p align="center"> <img src="./img1/equation3.png" alt="MLE" width="20%" height="20%"/> </p> 

# What is KL-divergence and ELBO?
KL divergence is defined like that
<p align="center"> <img src="./img1/equation4.png" alt="MLE" width="20%" height="20%"/> </p> 
It means metric function used to judge probability distribution Q and P is close. Lower KL-divergence value, closer they are. 

logP(X) is defined KL-divergece of Q(Z|X) and P(Z|X) plus ELBO term. logP(X) has limit, so higher ELBO value, lower KL-divergence. So ELBO term can become loss function.
<p align="center"> <img src="./img1/equation5.png" alt="MLE" width="20%" height="20%"/> </p> 

#
