# SAD Mini-Analysis

| SAD element | Student response |
|---|---|
| Problem statement | Academic risk may only become clear after a student receives poor results. This prototype explores whether historical student information can help identify records that may need additional academic support. |
| System users | Faculty member or academic analyst. |
| System inputs | Student attributes and academic indicators, including study time, absences, past failures, G1, and G2. The upload must contain the predictor columns used during training. |
| System process | Validate the uploaded CSV → preprocess the input using the saved pipeline → apply the trained Random Forest model → classify each record as AT RISK or NOT AT RISK → display and export the results. |
| System outputs | AT RISK / NOT AT RISK prediction labels, prediction counts, downloadable results, and a model evaluation summary generated during training. |
| Key limitation | The model is trained on one Portuguese secondary-school dataset and may not generalize to another institution or population. Predictions are estimates and require human review. |

## Mini use-case

A faculty member or academic analyst uploads a CSV file. The application checks whether required columns are present. If valid, the saved model predicts the risk category for each row. The application displays the results and provides a CSV download.
