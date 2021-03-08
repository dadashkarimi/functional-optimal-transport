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

# Entropy Regularization and Functional Optimal Transport

For paired time-series data from the same individual but from two different atlases according to the formulation in the paper:

<img src="https://render.githubusercontent.com/render/math?math=L_c(\mu_t,\nu_t) =\min_{{T}}C^T{T}\textbf{ s.t, }A\underline{{T}}=\begin{bmatrix}\mu_t\\ \nu_t \end{bmatrix}">

where:

![alt text](ot-eq.png)

The mapping ${T}$ represents the optimal way of transforming the brain activity data from $n$ regions into $m$ regions.

 Yet, solving a large linear program is computationally hard \cite{Dantzig:83}.
Thus, we use the  entropy regularization, which gives an approximation solution with complexity of  <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(n^2\log (n)\eta^{-3})"> for <img src="https://render.githubusercontent.com/render/math?math=\epsilon =\frac{4\log(n)}{\eta}">, and instead solve the following:

<img src="https://render.githubusercontent.com/render/math?math=L_c(\mu_t,\nu_t) =\min_{{T}}C^T {T} - \epsilon H(T)\textbf{ s.t, } A\underline{{T}}=\begin{bmatrix}\mu_t \\ \nu_t \end{bmatrix}">

 Specifically, we use the Sinkhorn algorithm---an iterative solution \cite{Altschuler:2017}---to find the optimum mapping ${T}$ as implemented in the Python Optimal Transport (POT) toolbox \cite{flamary2017pot}.

```python

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


all_268_mat = sio.loadmat('data/REST1/all_268.mat')
all_368_mat = sio.loadmat('data/REST1/all_368.mat')
all_268 = all_268_mat['data']
all_368 = all_368_mat['data']
from numpy import genfromtxt
all_behav = genfromtxt('data/REST1/IQ.txt', delimiter=',')
all_sex = genfromtxt('data/REST1/gender.txt', delimiter=',')

test_size = 200
train_size = all_268.shape[0]-test_size
random_samples = random.sample(range(0, all_268.shape[0]), all_268.shape[0])
train_index = random_samples[0:train_size]
test_index = random_samples[train_size:]

coord_268 =pd.read_csv('data/REST1/shen_268_matrix.csv', sep=',',header=None)
coord_368 = pd.read_csv('data/REST1/shen_367_matrix.csv', sep=',',header=None)
M = ot.dist(coord_268, coord_368,metric='cityblock')
M /= M.max()
```


# Parameter Tuning
As a control we run a couple of other experiments to verify usefulness of the new connectomes in brain-behavior association and individual classification.
To this aim we partition our data into three folds <img src="https://render.githubusercontent.com/render/math?math=g_1, g_2,"> and <img src="https://render.githubusercontent.com/render/math?math=g_3"> with respective ratio of \{0.25,0.5,0.25\}. 
We first train optimal transport mapping  T using all pairs of <img src="https://render.githubusercontent.com/render/math?math=\mu"> and <img src="https://render.githubusercontent.com/render/math?math=\nu"> in <img src="https://render.githubusercontent.com/render/math?math=g_1">. 
Then we apply T on all <img src="https://render.githubusercontent.com/render/math?math=\mu"> in <img src="https://render.githubusercontent.com/render/math?math=g_3"> to get high resolution <img src="https://render.githubusercontent.com/render/math?math=\nu"> (i.e., here from <img src="https://render.githubusercontent.com/render/math?math=268 \rightarrow 368">).
At the same time, we train a predictive model on all functional connectomes <img src="https://render.githubusercontent.com/render/math?math=F_{\nu} in g_2"> (i.e., <img src="https://render.githubusercontent.com/render/math?math=F_{\nu}\in\mathbb{R}^{368\times 368}">). 
At the end, we run the model on all functional connectomes obtained from optimal transport named <img src="https://render.githubusercontent.com/render/math?math=F^{T}_{\nu}\in g_3"> (i.e., test).
Our baseline is to test the same model on actual functional connectomes <img src="https://render.githubusercontent.com/render/math?math=F_{\nu}">.
We used fluid intelligence to study brain-behavior association and sex to classify participants based on.
We also tested significance of the results using re-sampled ttest. 

