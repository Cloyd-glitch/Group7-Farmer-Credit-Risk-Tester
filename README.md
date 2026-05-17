Agricultural lending institutions often face difficulties in evaluating farmer loan applicants because farming income is highly affected by crop yield, farming experience, debt level, irrigation access, and repayment behavior. Traditional loan evaluation methods are often manual, time-consuming, and inconsistent, which can increase the risk of loan defaults and financial losses.

This project proposes a Farmer Credit Risk Classification System using the ID3 (Iterative Dichotomiser 3) Decision Tree Algorithm to help classify farmer loan applicants into High Risk or Low Risk categories. The system uses entropy and information gain calculations to determine the most influential factors affecting farmer credit behavior and automatically generate interpretable decision rules.

The project utilizes a dataset containing 1,000 farmer loan applicant records collected from Kabacan, North Cotabato. Seven agricultural and financial features were analyzed, including loan repayment history, average crop yield, farming experience, farm size, debt amount, irrigation access, and crop type.

The ID3 algorithm recursively splits the dataset based on the feature with the highest information gain until the decision tree reaches optimal classification paths. The resulting model provides understandable and transparent decision-making compared to black-box machine learning approaches.

Several experiments were conducted to validate the performance of the proposed model, including tree depth control, cross-validation testing, feature importance analysis, and entropy versus Gini impurity comparison. Experimental results showed that the model achieved strong predictive performance with 84.5% testing accuracy while maintaining interpretability.

This project demonstrates how machine learning and decision tree analysis can support agricultural lending institutions in making faster, more accurate, and data-driven credit risk assessments for farmers.
