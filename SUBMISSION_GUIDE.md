# Final Submission Guide

This guide follows the six-item submission checklist in the laboratory handout.

1. **SAD mini-analysis:** submit `SAD_mini_analysis.md`.
2. **Python source files:** submit `train_model.py` and `app.py`.
3. **Dependencies and documentation:** submit `requirements.txt` and `README.md`.
4. **Model evaluation results:** after training, use the actual values in `model_evaluation.json` and `classification_report.txt`. Copy them into your report only after the script has run.
5. **Screenshots:** capture the real Dashboard page and a real Prediction Results page after running the Streamlit app. Do not use mock screenshots.
6. **GitHub URL:** after creating and pushing the public repository, replace the placeholder below with the real URL.

GitHub repository URL: `ADD YOUR ACTUAL REPOSITORY URL HERE`

## How to capture the two required screenshots

1. Run `streamlit run app.py`.
2. Open the Dashboard page and take a screenshot showing the metrics and charts.
3. Open Student Prediction, download the sample input CSV, and upload it.
4. Take a screenshot showing the prediction results and the predicted AT RISK count.
5. Save the images in a folder named `screenshots/` if your instructor requests the files in the repository.

## Important

The generated data, model, and metrics files do not exist until `python train_model.py` successfully retrieves the UCI data and trains the model. The GitHub repository URL and screenshots also require your own account and local application run. These items must be completed and verified before submission.
