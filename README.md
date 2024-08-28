# NLP Humor Classifier:
## Comparing Models and Vectorization Strategies for Text Classification

### Context

This analysis focuses on weighing the positives and negatives of different estimators and vectorization strategies for a text classification problem.  In order to consider each of these components, making use of the `Pipeline` and `GridSearchCV` objects in scikitlearn to try different combinations of vectorizers with different estimators.  For each of these, also using the `.cv_results_` to examine the time for the estimator to fit the data.

### Data

The dataset is from [kaggle](https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection) and contains a dataset named the "ColBert Dataset" created for this [paper](https://arxiv.org/pdf/2004.12765.pdf). Using the text column to classify whether or not the text was humorous. 

**Note:** The original dataset contains 200K rows of data. It is best to try to use the full dtaset. If the original dataset is too large for your computer, please use the 'dataset-minimal.csv', which has been reduced to 100K.

### Text preprocessing:

As a pre-processing step, performed both `stemming` and `lemmatizing` to normalize the text before classifying. For each technique used both the `CountVectorize`r and `TfidifVectorizer` and used options for stop words and max features to prepare the text data for the estimator.

### Classification Models:

Once the text data is prepared with stemming lemmatizing techniques, Used `LogisticRegression`, `DecisionTreeClassifier`, and `MultinomialNB` as classification algorithms for the data. Compared their performance in terms of accuracy and speed.

### Evaluation

<div align="center">
    <img src="https://github.com/mitbans/NLP-Humor-Classifier/blob/main/images/comparingtraintime.png" width="500" height="400" alt="Image 1">
    <img src="https://github.com/mitbans/NLP-Humor-Classifier/blob/main/images/comparinfgscores.png" width="500" height="400" alt="Image 2">
</div>


#### Logistic Regression 
Consistently outperform the others, especially when using stemming with count vectorization. These models offer the best balance of precision, recall, F1, and AUC, making them well-suited for text classification tasks that require accurate class identification and discrimination. Although stemming slightly improves metrics, it is more computationally expensive compared to lemmatizing.

#### Decision Trees 
Exhibits strong recall and F1 scores but fall short of logistic regression across all metrics.

#### Naive Bayes 
Delivered competitive performance with higher computational costs and lower metrics compared to logistic regression.

#### Conclusion
Overall, the `logistic_stem_count` model stands out due to its robust performance across all key metrics. However, if computational efficiency is a concern, the `logistic_lemmatize_count` model is a viable alternative.

## Repository Structure
- <code>text_data/dataset.csv</code>: Contains dataset used in the analysis.
- <code>images/</code>: Contain metrics comparison charts.
- <code>notebooks/NLP-Humor-Classifier.ipynb</code>: Jupyter notebook with code for data analysis.
- <code>README.md</code>: Summary of findings and link to notebook

## Notebook
The detailed analysis and code can be found in the Jupyter notebook <a href="[https://github.com/mitbans/NLP-Humor-Classifier/blob/main/notebooks/NLP-Humor-Classifier.ipynb">here</a>.

---

