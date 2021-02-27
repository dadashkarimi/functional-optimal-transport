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

# Parameter Tuning
As a control we run a couple of other experiments to verify usefulness of the new connectomes in brain-behavior association and individual classification.
To this aim we partition our data into three folds $g_1$, $g_2$, and $g_3$ with respective ratio of $\{0.25,0.5,0.25\}$. 
We first train optimal transport mapping $\pazocal T$ using all pairs of $\mu$ and $\nu$ in $g_1$. 
Then we apply $\pazocal T$ on all $\mu$ in $g_3$ to get high resolution $\nu$ (i.e., here from $268 \rightarrow 368$).
At the same time, we train a predictive model on all functional connectomes $F_{\nu}$ in $g_2$ (i.e., $F_{\nu} \in \mathbb{R}^{368\times 368}$). 
At the end, we run the model on all functional connectomes obtained from optimal transport named $F^{\pazocal T}_{\nu}$ in $g_3$ (i.e., test).
Our baseline is to test the same model on actual functional connectomes $F_{\nu}$.
We used fluid intelligence to study brain-behavior association and sex to classify participants based on.
We also tested significance of the results using re-sampled ttest. 

