# Active-Learning-for-gas-mixture-adsorption-prediciton
![Krish_AL_mixtures_workflow](https://user-images.githubusercontent.com/36941306/223895808-1b404ce3-b044-44fe-be13-f31a2f252ccc.png)

Active learning workflow for predicting adsorption using gaussian process regression (GPR). 
The learning starts from pre-processing the prior data. Pressure and temperature are standardised, while the mole-fraction is linearly scaled to –1 and 1, i.e. X* = (X - 1/2)x25/12. Then it is passed through the dual-GPs, one for each species. Then prediction are done, and the associated uncertainties are extracted. The accuracies for both the species are tested for convergence. If any of the accuracy criteria is not met, learning continues, and the point with the highest uncertainty is added to the prior data. The active learning continues until the convergence condition is satisfied.
The Accuracy parameter is defined as follows:
```math
     ACC_{i} = 100 \times \frac{X_{+}}{X_{+}+X_{-}}
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
Also, the $\beta$ has a value of 2% for P-X and 5% for P-X-T calculation.$\{sigma}_{n_{i}}$ and ${y}^{'}_{n_{i}}$ are the uncertainty and adsorption value (scaled) associated with the test point $X_{n_{i}}$. The threshold value $\beta_{i}$ is user-defined and can be set on the basis of the desired confidence the user needs. Also, we have kept the $\beta$ values same for the all the species in the three gas mixtures. We had $\beta$ set to 2\% for the P–X phase space, while we had chosen a high upper limit of 5\% for active learning in the P–X–T space.
