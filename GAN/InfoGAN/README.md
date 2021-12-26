# Summary
In this paper, rather than using a single unstructured noise vector, we propose to decompose the input
noise vector into two parts: (i) z, which is treated as source of **incompressible noise;** (ii) c, which we
will call the latent code and will target the salient structured **semantic features of the data distribution**

For ease of notation, we will use latent codes c to denote the concatenation of all latent variables c_i.

**This paper's goal is to suggest the method of discovering these latent factors in an unsupervised way**

we provide the generator network with both the incompressible noise z and the latent code c, so the form of the
generator becomes G(z, c). If we use the standard form of GAN, the the generator is free to ignore the additional
latent code c by finding a solution satisfying P_G(x|c) = P_G(x). It means generator can generate fixed image.

To solve this problem, it suggests an information-theoretic regularization: there should be high mutual information
between **latent codes c and generator distribution G(z, c).** Thus I(c; G(z, c)) should be high. If mutual information is high, c is dependent on G(z,c). So we can know about G(z, c)'s information by using latent code c




# Informaton theory
## entropy

## conditional entropy

## joint entropy

## kl divergence
- It measures two distribution's shape and support is close or not .
- For example, P(x) and Q(x)'s shape and support is closer, kl-divergence's value is lower.
- It is calculated, 
<p align="center"> <img src="./img/INFOGAN_kldivergence.png" alt="MLE" width="40%" height="40%"/> </p>

- Note: KL(Q||P) is not equal to KL(P||Q). So it doesn't satisfy distance function's property.

## mutual information
- It means the method of measuring how much mutual dependency two random variables are. 
- If random variable x and y is independent, joint distribution p(x,y) is equal to p(x)p(y)
- Mutual information's formula is equal to KL(P(x,y) || P(X)P(Y))
- In this equation, we can infer that, P(X,Y) = P(X)P(Y), kl-divergence is 0, so X and Y are independent.
- We can measure how much dependent two random variable are, and how much they share mutual information, so we call it mutual information. 
- Mutual information : I(x,y) = KL(p(x,y)||p(x)p(y)) = H(x) - H(x|y) = H(y) - H(y|x)  
<p align="center"> <img src="./img/INFOGAN_mutualinformation.png" alt="MLE" width="70%" height="70%"/> </p>
H(X) = X의 불확실한 정도. (=Y가 주어지지 않을 때 x의 불확실한 정도)
H(X|Y) = Y가 주어졌을 때, X의 불확실한 정도.
H(X) - H(X|Y) = Y로 인해 알아낸 X의 정보량.

if) X and Y is independent, H(X) - H(X|Y) = H(X) - H(X) = 0, because, if you know Y but you don't infer any information of X by using known Y.
H(Y|X)'s value get lower accroding to how dependent X and Y are. So I(x;y)'s value get bigger because of I(x,y) is equal to H(y) - H(y|x).


# Loss function
<p align="center"> <img src="./img/CGAN_lossfunction.png" alt="MLE" width="70%" height="70%"/> </p>

# Model
## Network design


## Network reference

# Results

# Problem

# Reference
https://ratsgo.github.io/statistics/2017/09/22/information/
https://hyunw.kim/blog/2017/10/14/Entropy.html
