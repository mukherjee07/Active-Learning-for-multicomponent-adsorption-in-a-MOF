# Active Learning for gas mixture adsorption prediction in a MOF #
## Files, Folders, and Description ##
[`Adsorption isotherm and Error map plots`](Adsorption isotherm and Error map plots) folder: After Active learning is complete, these plots can be used to visualize the adsorption isotherm and error heat map for the final GP fits. All the data files are provided for visualization.

[`P-X`](P-X) folder: Active learning code for adsorption prediction in P-X phase space for 3 gas mixtures.
<ul>
  <li>training.csv and complete.csv are the training data and ground truth data for the systems. The training.csv provided in these folders are the initial training set as defined in the paper. As the AL algorithm start, it will update the next set of training points in this file only.</li>
  <li>mean.csv is the datafile that will be generated when the AL algorithm starts. This datafile contains the GP-MRE for both the species, the maximum GP relative errors, the MREs, $R^2$'s, perceived accrucies (with $\beta$ value of 2% as well as 5% for P-X-T, while 1% and 2% for P-X, for comparision purposes).</li>
  <li>GP_mixtures.py is the Active learning engine (with dual-GP model)</li>
  <li>Active-learning-sh is the model updation handler which works in linux environment. This was based on Notre Dame Center for computing resource which is a grid-engine system. One can modify the first few lines depending on the specific linux environment or to run it locally, the first line can be removed as well.</li>
</ul>

[`P-X-T`](P-X-T) folder: Active learning code for adsorption prediction in P-X-T phase space for 3 gas mixtures.
Same process as P-X phase space.

[`Kernel_opt`](Kernel_opt) folder: This includes the Kernel optimization python code with the datafiles for three gases for P-X and P-X-T cases having different kernel combination upto 500 iterations (39 for P-X-T and 12 for P-X).

[`raspa2_May_2018`](raspa2_May_2018) folder: This is the submodule of the raspa version which was used to generate the ground truth data. This version is of May 8 2018. It was originally cloned from the University of Amsterdam github link (original developers are David Dubbledam and co.).
 
## System ##
The goal was to predict the adsorption isotherm of a binary mixture adsorption in a Cu-BTC MOF (also known as HKUST-1) using an Active learning protocol.

We investigated three different gas mixtures (CO<sub>2</sub>-CH<sub>4</sub>, Xe-Kr, and H<sub>2</sub>S-CO<sub>2</sub>) at different pressure, temperature and mole fraction conditions (thus 2 different studies). The low pressure limit was same for all mixtures ($10^{-6}$ bar) but the high pressure limit was different. The temperaure (200-400K) and mole-fraction (0.02-0.98) range were same. This paper is soon to be submitted.


<img width="1227" alt="Screen Shot 2023-03-09 at 1 24 16 AM" src="https://user-images.githubusercontent.com/36941306/223938253-d6953813-48cd-44b8-8aee-1f82f98f2d53.png">

## Ground-truth ##
Ground-truth data was generated using grand-canonical Monte Carlo (GCMC) simulations and were performed in the open-source software RASPA. Forcefield for CO<sub>2</sub>, CH<sub>4</sub>, Xe, Kr, and H<sub>2</sub>S used were TraPPE and for Cu-BTC it was Universal forcefield (UFF). Also, component-fractional Monte Carlo (CFC-GCMC) method was used to sample the GCMC simulation. All the ground truth data can be found in the complete.csv in the P-X and P-X-T folders.

## Algorithm ##
A general workflow outline is shown below:

![AL_Outline](https://github.com/mukherjee07/Active-Learning-for-multicomponent-adsorption-in-a-MOF/assets/36941306/fba29e4b-aa04-476b-ac34-b25fc73a0013)


Active learning workflow for predicting adsorption using gaussian process regression (GPR). 
The learning starts from pre-processing the prior data. Pressure and temperature are standardised, while the mole-fraction is linearly scaled to –1 and 1, i.e. X* = (X - 1/2)x25/12. Then it is passed through the dual-GPs, one for each species. Then prediction are done, and the associated uncertainties are extracted. The perceived accuracies (PAC) for both the species are tested for convergence. If any of the PAC criteria is not met, learning continues, and the point with the highest uncertainty is added to the prior data. The active learning continues until the convergence condition is satisfied.
The PAC parameter is defined as follows:

```math
     PAC_{i} = 100 \times \frac{X_{+}}{X_{+}+X_{-}}
```
```math
\begin{aligned}
     \text{If at $X_{n_{i}}$,} \;
     |\frac{\sigma_{n_{i}}}{y^{'}_{n_{i}}}| <= \beta_{i}, \;
     \text{then,} \; X_{+} = X_{+} + 1 \;
     \text{else,} \;
      X_{-} = X_{-} + 1
      \end{aligned}
 ```
Also, the $\beta$ has a value of 2% for P-X and 5% for P-X-T calculation. $\sigma_{n_{i}}$ and  $y_{n_{i}}$ are the uncertainty and adsorption value (scaled) associated with the test point $X_{n_{i}}$. The threshold value $\beta_{i}$ is user-defined and can be set on the basis of the desired confidence the user needs. Also, we have kept the $\beta$ values same for the all the species in the three gas mixtures. We had $\beta$ set to 2\% for the P–X phase space, while we had chosen a high upper limit of 5\% for active learning in the P–X–T space.

## Key Performance parameters ##

Mean Relative Error (MRE):

```math
    \text{MRE in \%} = \left(\sum_{i=1}^{n} \Bigg|{\frac{Y_{\text{GP-predict}}(x_i) - Y_{\text{GCMC}}(x_i)}{Y_{\text{GCMC}}(x_i)+\boldsymbol{\epsilon}}}\Bigg|\right) \times \frac{100}{n}
```
The $\boldsymbol{\epsilon}$ (= $10^{-3}$) is added to the denominator to avoid numerical issues since adsorption in some phase spaces can reach 0. 

Data requirement (% of ground truth):
```math
    \text{Data Requirement in \%} = \frac{{N_{Initial-training}} + {N_{Iterations-to-90\% PAC}}}{{N_{Ground-truth}}}\times100
```
Note $N_{Prior}$ for P-X and P-X-T were constant at 54 and 90, respectively. While $N_{Ground-truth}$ was 2499 (51P x 49X) for P-X and 2541 (21P x 11X x 11T) for P-X-T.

R<sup>2</sup> (The coefficient of Determination):
```math
R^2(Y_{\text{GCMC}}, Y_{\text{GP-predict}}) = 1 - \frac{{\sum_{i=1}^{n}(Y_{\text{GCMC}}(x_i) - Y_{\text{GP-predict}}(x_i))^2}}{{\sum_{i=1}^{n}}{(Y_{\text{GCMC}}(x_i) - \overline{Y_{\text{GCMC}}(x_i)})^2}}
```

## Results ##

The performance of the Active learning workflow is below for both the phase spaces and three systems:

Results for the P-X phase space:

| Mixture | Kernel | Data requirement (% of ground truth) | MRE<sub>(species 1)</sub> (%) | MRE<sub>(species 2)</sub> (%) | R<sup>2</sup><sub>(species 1)</sub> | R<sup>2</sup><sub>(species 2)</sub> |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CO<sub>2</sub>-CH<sub>4</sub> | RBF | 3.001 | 5.263 | 5.417 | 0.986 | 0.999 |
| Xe-Kr | RQ | 2.601 | 6.526 | 6.394 | 0.985 | 0.998|
| H<sub>2</sub>S-CO<sub>2</sub> | RQ | 2.561 | 7.149 | 7.154 | 0.982 | 0.995 |


Results for the P-X-T phase space:

| Mixture | Kernel | Data requirement (% of ground truth) | MRE<sub>(species 1)</sub> (%) | MRE<sub>(species 2)</sub> (%) | R<sup>2</sup><sub>(species 1)</sub> | R<sup>2</sup><sub>(species 2)</sub> |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CO<sub>2</sub>-CH<sub>4</sub> | RBF+RBF+RBF | 6.611 | 5.461 | 9.256 | 0.988 | 0.990 |
| Xe-Kr | RBF+RBF+RBF | 6.650 | 4.850 | 7.025 | 0.990 | 0.990|
| H<sub>2</sub>S-CO<sub>2</sub> | RQ | 5.549 | 8.276 | 11.682 | 0.976 | 0.986 |

