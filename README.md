# Functional Optimal Transport

Here we propose to use optimal transport, a traditional mathematical technique, to find optimum transportation policies between connectomes in different resolutions. A policy then is used to transport time series between parcellations given the geometry of brain atlases.
We evaluate the proposed method by measuring the similarity between connectomes obtained by optimal transports and analogous connectomes in training.

![alt text](fig-1.png)

# Optimal Transport 

<img src="https://render.githubusercontent.com/render/math?math=    \min_{\pazocal T} \Big\{ \sum_i c(x_i,\pazocal T (x_i)) : \pazocal T_{\sharp} \alpha = \beta \Big\},  = -1">

