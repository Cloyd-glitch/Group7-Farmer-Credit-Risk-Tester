# Group7-Farmer-Credit-Risk-Tester
Agricultural lending institutions often face challenges in evaluating farmer loan applicants because farming income is affected by factors such as crop yield, farming experience, debt level, irrigation access, and repayment history. Manual loan assessment methods can be time-consuming and may lead to inconsistent credit decisions.

This project proposes a Farmer Credit Risk Classification System using the ID3 (Iterative Dichotomiser 3) Decision Tree Algorithm to classify farmers into High Risk or Low Risk categories. The system analyzes agricultural and financial data from 1,000 farmer loan applicant records collected in Kabacan, North Cotabato.

Using entropy and information gain, the ID3 algorithm identifies the most influential factors affecting credit risk and generates interpretable decision rules. Experimental results showed that the model achieved strong predictive performance with an accuracy of 84.5%, demonstrating its potential to support data-driven agricultural loan evaluation.

**Project Information**
  -Dataset Size: 1,000 farmer loan applicant records
  -Features: 7 predictor variables
  -Target Variable: credit_risk {High, Low}
  -Algorithm: ID3 Decision Tree
  -Splitting Criterion: Entropy and Information Gain
  -Maximum Tree Depth: 4

**Dataset Features**
Feature	                               Type	                                  Description
loan_repayment_hist	                   Binary	                                Farmer repayment history
avg_yield_tons_ha	                     Continuous	                            Average crop yield per hectare
years_experience	                     Continuous	                            Farming experience in years
farm_size_ha	                         Continuous	                            Total farm size in hectares
current_debt_php	                     Continuous	                            Existing debt amount
irrigation_access	                     Binary	                                Availability of irrigation
crop_type                              Categorical	                          Type of crop cultivated
credit_risk	                           Target	                                High Risk / Low Risk

**ID3 Decision Tree Model**
The ID3 Decision Tree Algorithm builds the classification tree using:
  -Entropy
  -Information Gain
  -Recursive dataset splitting
  -Majority class labeling

Entropy Formula
H(S) = - Σ p(x) log₂ p(x)
Information Gain Formula
IG(S, f) = H(S) − Σ (|Sv| / |S|) · H(Sv)

**Root Node Result**
Feature                              	Information Gain
loan_repayment_hist	                      0.1984
avg_yield_tons_ha	                        0.0438
years_experience	                        0.0414
farm_size_ha	                            0.0359
current_debt_php	                        0.0337
irrigation_access	                        0.0211
crop_type	                                0.0139

**Root Decision Node**
  -loan_repayment_hist
This feature generated the largest entropy reduction and was identified as the strongest predictor of farmer credit risk.

**Experiment A — Tree Depth Control**
max_depth	                      Train Accuracy	              Test Accuracy
2	                                  75.12%	                      78.00%
3	                                  78.75%	                      82.50%
4          	                        82.37%	                      84.50%
5	                                  85.25%	                      80.00%

**Experiment B — Cross Validation**
Evaluation                       Method	                      Mean Accuracy
80/20                            Split	                        84.50%
5-Fold                           CV	                            79.60%
10-Fold                          CV	                            80.20%

**Experiment C — Feature Importance**
Feature	                      Importance Score
loan_repayment_hist              	0.4490
avg_yield_tons_ha	                0.1740
farm_size_ha	                    0.1589
years_experience	                0.1050
current_debt_php	                0.0780
irrigation_access	                0.0351
crop_type                        	0.0000

**Experiment D — Entropy vs Gini Impurity**
Criterion	            Test Accuracy
Entropy (ID3)	          84.50%
Gini Impurity	          79.00%

**Model Performance**
Metric	                           Result
Accuracy	                          84.5%
High Risk Precision	                89.66%
High Risk Recall	                  78.00%
High Risk F1-Score	                83.42%
Low Risk Precision	                80.53%
Low Risk Recall	                    91.00%
Low Risk F1-Score	                  85.45%






