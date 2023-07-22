# Sentimental Analysis on Twitter Tweets 

The goal of this project was to perform sentiment analysis on Twitter data to classify tweets as positive, negative or neutral. The project involved:

* Data Collection: Twitter data was collected using the Twitter API. Tweets containing specific keywords over a defined time period were extracted.

* Data Preprocessing: The raw Twitter data was cleaned by removing URLs, usernames, hashtags, special characters etc. Tokenization and lemmatization was applied to extract meaningful words and sentences from tweets.

* Sentiment Analysis: The preprocessed tweets were analyzed using the VADER sentiment analysis tool to assign positive, negative or neutral sentiment scores.

* Model Building: A logistic regression model was trained on labeled sentiment data to validate the performance of the VADER sentiment analysis.

* Evaluation: The VADER sentiment analyzer achieved 85% accuracy in classifying tweet sentiment when evaluated against the logistic regression model.

The project provided hands-on experience with sentiment analysis on unstructured Twitter data using natural language processing techniques. The resulting model can analyze public sentiment on various topics across Twitter to gain valuable insights.

## Big_Data Concepts used in this project

* Distributed Computation - The project leveraged distributed computation frameworks like MapReduce to process large volumes of Twitter data efficiently in parallel. This enables scalable analysis.

* Data Ingestion - The Twitter API was used to collect streaming tweet data which is an example of ingesting semi-structured data from an online source.

* Data Cleaning - Techniques like tokenization, removal of stop words, lemmatization, etc. were used to clean the noisy raw Twitter data before analysis.

* Natural Language Processing - Sentiment analysis involves applying NLP techniques like text normalization, part-of-speech tagging, named entity recognition, etc. to extract insights.

* Machine Learning - Supervised learning algorithms like logistic regression were applied to the text data for sentiment classification modeling.

* Model Evaluation - Evaluation metrics like accuracy, precision, recall, etc. were used to validate the performance of machine learning models.

* Visualization - Data visualizations can be used to understand and present the sentiment analysis results effectively.

* Streaming Analysis - For real-time applications, streaming analysis of live tweets can be done to continuously monitor sentiment.

## Languages and Libraries used in this sentiment analysis are

* Python - Python was the core programming language used to implement the data collection, preprocessing, modeling and analysis.

* PySpark - PySpark, the Python API for Spark, was used for distributed data processing and analysis using Spark's distributed computing capabilities.

* tweepy - The tweepy library was used in Python to access the Twitter API and collect tweet data.

* NLTK - The Natural Language Toolkit (NLTK) Python library was used for text processing and natural language processing tasks.

* VADER - The Valence Aware Dictionary and sEntiment Reasoner (VADER) sentiment analysis tool from NLTK was used for analyzing sentiment.

* Logistic Regression - Scikit-learn's implementation of logistic regression in Python was used as the machine learning algorithm.

* Spark SQL - Spark SQL was used on top of PySpark for some data processing tasks like aggregations.

* Matplotlib - Matplotlib Python library was used for some visualization tasks.

So overall, the core tech stack used was:

1. Python for programming

2. PySpark for distributed processing

3. NLTK and VADER for NLP

4. Scikit-learn for ML modeling

5. SQL and Matplotlib for analysis and vis

This demonstrates how Python's extensive data analytics and ML libraries can be leveraged for big data pipelines.
