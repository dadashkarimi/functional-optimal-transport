# Functional Optimal Transport

Here we propose to use optimal transport, a traditional mathematical technique, to find optimum transportation policies between connectomes in different resolutions. A policy then is used to transport time series between parcellations given the geometry of brain atlases.
We evaluate the proposed method by measuring the similarity between connectomes obtained by optimal transports and analogous connectomes in training.

![alt text](fig-1.png)

# Monge Problem

<img src="https://render.githubusercontent.com/render/math?math=\min_{T}\Big\{, \sum_i c(x_i,T(x_i)):T_{\sharp}\alpha =\beta\Big\}">
# Kantorvic Relaxation
Kantorvich rather solves the mass transportation problem using a probabilistic approach in which the amount of mass located at $x_i$ potentially dispatches to several points in target \cite{Kantorovich1:42}.  
Admissible solution for Kantorvich relaxation is defined by a coupling matrix $\pazocal{T} \in {R}^{n\times m}_+$ indicating the amount of mass being transferred from location $x_i$ to $y_j$ by ${T}_{i,j}$:

<img src="https://render.githubusercontent.com/render/math?math=U(a,b)=\{{T}\in\mathbb{R}^{n\times m}_+:{T}{1}_m =a,{T}^T{1}_n=b\},">

