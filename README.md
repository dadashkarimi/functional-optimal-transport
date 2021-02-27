# Functional Optimal Transport

Here we propose to use optimal transport, a traditional mathematical technique, to find optimum transportation policies between connectomes in different resolutions. A policy then is used to transport time series between parcellations given the geometry of brain atlases.
We evaluate the proposed method by measuring the similarity between connectomes obtained by optimal transports and analogous connectomes in training.

![alt text](fig-1.png)

# Monge Problem

<img src="https://render.githubusercontent.com/render/math?math=\min_{T}\Big\{, \sum_i c(x_i,T(x_i)):T_{\sharp}\alpha =\beta\Big\}">

<img src="https://render.githubusercontent.com/render/math?math=U(a,b)=\{{T}\in\mathbb{R}^{n\times m}_+:{T}\mathbbm{1}_m =a,{T}^T\mathbbm{1}_n=b\},">

