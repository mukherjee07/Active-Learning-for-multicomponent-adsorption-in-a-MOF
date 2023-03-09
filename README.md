# Active-Learning-for-gas-mixture-adsorption-prediciton
![Krish_AL_mixtures_workflow](https://user-images.githubusercontent.com/36941306/223895808-1b404ce3-b044-44fe-be13-f31a2f252ccc.png)

Active learning workflow for predicting adsorption using gaussian process regression (GPR). 
The learning starts from pre-processing the prior data. Pressure and temperature are standardised, while the mole-fraction is linearly scaled to –1 and 1, i.e. X* = (X - 1/2)x25/12. Then it is passed through the dual-GPs, one for each species. Then prediction are done, and the associated uncertainties are extracted. The accuracies for both the species are tested for convergence. If any of the accuracy criteria is not met, learning continues, and the point with the highest uncertainty is added to the prior data. The active learning continues until the convergence condition is satisfied.
```math
     ACC_{i} = 100 \times \frac{X_{+}}{X_{+}+X_{-}}
```

