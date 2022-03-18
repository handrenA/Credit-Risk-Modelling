# Credit-Risk-Modelling
Analyzing german bank customers if they have good/bad credit risk to the bank by analyzing the dataset with visualizations. Then proceeding 
with doing k-cluster means to find if there's any distinct groups among the customers. 4 distinct groups were found, majority of the groups are found
in the good-rating ratio and the two other groups are found in the bad-rating ratio.

### Model selection 
Different classifications algorithms were tested and Random Forest was the one that turnet out to be most suited one for this problem 
after doing hyperparamater tuning with RandomizedSearch.

### Metric
Since we want to predict all of the good-rating customers _and_ be very sure about it - the F1-score is the most optimal to go with here. 
Also the dataset was kind of inbalanced (70% good/30% bad) so F1 suits very well. Score: 0.84.


