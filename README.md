# Functional Optimal Transport

Here we propose to use optimal transport, a traditional mathematical technique, to find optimum transportation policies between connectomes in different resolutions. A policy then is used to transport time series between parcellations given the geometry of brain atlases.
We evaluate the proposed method by measuring the similarity between connectomes obtained by optimal transports and analogous connectomes in training.

![alt text](fig-1.png)

# Monge Problem
Lets define some locations $x_1,..,x_n$ in $\alpha$ and some locations $y_1,..,y_m$ in $\beta$. Then we specify weight vector $a$ and $b$ over these locations and define matrix $C$ as a measure of pairwise distances between points $x_i \in \alpha$ and comparable points $\pazocal T (x_i)$;
Monge problem aims to solve the following optimizatin problem:

<img src="https://render.githubusercontent.com/render/math?math=\min_{T}\Big\{, \sum_i c(x_i,T(x_i)):T_{\sharp}\alpha =\beta\Big\}">

# Kantorvic Relaxation

Kantorvich rather solves the mass transportation problem using a probabilistic approach in which the amount of mass located at <img src="https://render.githubusercontent.com/render/math?math=x_i"> potentially dispatches to several points in target.  
Admissible solution for Kantorvich relaxation is defined by a coupling matrix <img src="https://render.githubusercontent.com/render/math?math=T\in{R}^{n\times m}"> indicating the amount of mass being transferred from location <img src="https://render.githubusercontent.com/render/math?math=x_i"> to <img src="https://render.githubusercontent.com/render/math?math=y_j"> by <img src="https://render.githubusercontent.com/render/math?math=T_{i,j}">:

<img src="https://render.githubusercontent.com/render/math?math=U(a,b)=\{{T}\in\mathbb{R}^{n\times m}_+:{T}{1}_m =a,{T}^T{1}_n=b\},">

