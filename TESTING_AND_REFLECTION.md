# Testing and Reflection Record

## Testing checklist

Mark each item only after personally running and observing the expected result. Do not mark PASS based only on reading the code.

| Test | Procedure | Expected result | Actual result / evidence | Status |
|---|---|---|---|---|
| Run `train_model.py` | Activate the environment and run `python train_model.py`. | `student_data.csv`, `student_model.pkl`, and evaluation output files are created without errors. | Record terminal output and metric values. | NOT YET RUN |
| Run Streamlit app | Run `streamlit run app.py`. | Dashboard opens in the browser. | Capture dashboard screenshot. | NOT YET RUN |
| Upload valid sample CSV | Download the sample input from the app, then upload it. | Predictions appear in a results table. | Capture prediction results screenshot. | NOT YET RUN |
| Upload CSV with missing required column | Remove one predictor column from a copy of the sample and upload it. | App displays a helpful missing-column error. | Record the displayed message. | NOT YET RUN |
| Download prediction results | Click the results download button after a valid upload. | A CSV containing the prediction column is downloaded. | Open the downloaded CSV and confirm its columns. | NOT YET RUN |
| Check GitHub repository | Open the public repository URL after pushing. | Source files and README are visible. | Paste the actual repository URL. | NOT YET RUN |

## Reflection questions

### 1. What is the prediction target, and why is G3 excluded from the input features?

The target is whether a student is AT RISK or NOT AT RISK based on the final Mathematics grade. A G3 below 10 is AT RISK, while a G3 of 10 or higher is NOT AT RISK. G3 is excluded from the input because it is the value we want the model to estimate. If it were included, the model could simply use the answer while predicting, which is called target leakage.

### 2. What does the model's recall for the 'At risk' class tell you?

Recall for the AT RISK class tells us how many of the students who are actually at risk in the test data were correctly identified by the model. A higher recall means fewer at-risk records were missed. It is important to check this metric because missing a student who may need support can matter.

### 3. Why might a model trained on this dataset perform differently on students from your institution?

The dataset came from two secondary schools in Portugal. Our institution may have different students, learning environments, grading practices, resources, and academic support systems. Because of these differences, the patterns learned by the model may not fit our students as well.

### 4. What action should a faculty member take after seeing an 'AT RISK' prediction?

The faculty member should review the student's actual situation and talk with the student in a supportive way. They can check attendance, learning difficulties, and other relevant information, then offer appropriate academic support. The prediction should not be treated as a final decision or label for the student.
