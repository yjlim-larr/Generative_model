# Summary
In this paper, rather than using a single unstructured noise vector, we propose to decompose the input
noise vector into two parts: (i) z, which is treated as source of **incompressible noise;** (ii) c, which we
will call the latent code and will target the salient structured **semantic features of the data distribution**

For ease of notation, we will use latent codes c to denote the concatenation of all latent variables c_i.

**This paper's goal is to suggest the method of discovering these latent factors in an unsupervised way**

we provide the generator network with both the incompressible noise z and the latent code c, so the form of the
generator becomes G(z, c). If we use the standard form of GAN, the the generator is free to ignore the additional
latent code c by finding a solution satisfying P_G(x|c) = P_G(x). It means generator can generate fixed image.

To solve this problem, it suggests an information-theoretic regularization:

# Informaton theory
## entropy

## kl divergence

## cross entropy

## joint entropy

## conditional entropy 

## mutual information

# Loss function
<p align="center"> <img src="./img/CGAN_lossfunction.png" alt="MLE" width="70%" height="70%"/> </p>

# Model
## Network design


## Network reference

# Results

# Problem
a
# Reference
