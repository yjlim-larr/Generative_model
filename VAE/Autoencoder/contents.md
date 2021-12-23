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

<p align="center"> <img src="./img1/equation3.png" alt="MLE" width="70%" height="70%"/> </p> 

# What is KL-divergence and ELBO?
KL divergence is defined like that
<p align="center"> <img src="./img1/equation4.png" alt="MLE" width="40%" height="40%"/> </p> 
It means metric function used to judge probability distribution Q and P is close. Lower KL-divergence value, closer they are. 

logP(X) is defined KL-divergece of Q(Z|X) and P(Z|X) plus ELBO term. logP(X) has limit, so higher ELBO value, lower KL-divergence. So ELBO term can become loss function.
<p align="center"> <img src="./img1/equation5.png" alt="MLE" width="70%" height="70%"/> </p> 
<p align="center"> <img src="./img1/equation6.png" alt="MLE" width="40%" height="40%"/> </p> 
if 1) has high value, it has high probability which decoder ouptut same image to encoder input. 1) is called "reconstruction error"
2) means "regularization"

loss function is like that
<p align="center"> <img src="./img1/equation7.png" alt="MLE" width="20%" height="20%"/> </p> 

# Calculate loss function gradient
using monte carlo gradient estimator for caculating loss function gradient. For example 
<p align="center"> <img src="./img1/equation8.png" alt="MLE" width="20%" height="20%"/> </p> 

And for making differentiable funciton, transform q(z|x).
<p align="center"> <img src="./img1/equation9.png" alt="MLE" width="20%" height="20%"/> </p> 
Because, q(z|x) is not differentiable. 



# Train method
## Train order

## reparameterization trick
