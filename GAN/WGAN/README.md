# WGAN
paper link: https://arxiv.org/pdf/1701.07875.pdf

# Summary
To solve KL divergence problem, insert noise to model distribution. if it doesn'y, KL divergence diverge to infinite.
Therefore two probability distribution's disjoint support's probability should have not zero probability by inserting noise.  
But this method make image blurry. Output pixel is normalized to [0, 1] range, therefore small noise makes image blurry.  
Gan solve these problems by mapping p(z) to p_data(x) directly. 하지만 GAN도 문제점이 많이 존재한다. 

This paper suggest new metric functions, and using this function gurantee P_generator(x) converge to P_real(x). model distribution의 수렴은 분포의 거리함수를 어떻게 정의하는가와 
관련이 있다. 거리 함수가 연속이 되기 쉽고, 수렴이 되기 쉬울수록 신경망의 모수와 model distribution을 mapping하는 함수 f을 쉽게 연속으로 정의할 수 있다. 즉, 쉽게 분포를 수렴시킬 수 있다. 
Loss 함수가 연속적이면, 해당 loss function을 이용하여 신경망을 학습시킬 때 신경망의 모수와 model distribution을 mapping 시키는 함수 f가 연속이 되도록 loss 함수를 정의할 것이다. 왜냐하면, 
loss function으로 부터 신경망의 모수를 수렴시키면, 해당 모수가 model distribution을 수렴시키는 것이 보장되어야 하기 때문이다.

## 거리함수
1. total variatoin distance
2. KL divergence
3. JS divergence
4. EM distanve or Wasserstein-1
<p align="center"> <img src="./img/WGAN_dis.png" alt="MLE" width="50%" height="50%"/> </p>








