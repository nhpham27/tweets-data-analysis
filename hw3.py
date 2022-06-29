#%%
import numpy as np
import pandas as pd
import nltk
import sklearn 
import string
import re # helps you filter urls
from sklearn.metrics import accuracy_score
#%%
#Whether to test your Q9 for not? Depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    text = text.lower()
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]' \
                    + '|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = text.replace("'s", "")
    text = text.replace("'", "")
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, "")
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " ")
    tokens = nltk.word_tokenize(text)
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tags = nltk.pos_tag(tokens)
    new_tags = []
    for tag in tags:
        if posMapping.get(tag[1][0]) is not None:
            new_tags.append(lemmatizer.lemmatize(tag[0], posMapping[tag[1][0]]))
        else:
            new_tags.append(lemmatizer.lemmatize(tag[0], 'n'))
    return new_tags
    
#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
#     [YOUR CODE HERE]
    df['text'] = df['text'].apply(process)
    return df
    
#%%
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        processed_tweets: pd.DataFrame: processed tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords (after processing)
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
#     [YOUR CODE HERE]
    def same_text(text): 
        return text
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False,
                                                                 min_df=2,
                                                                 stop_words=stop_words,
                                                                 analyzer='word',
                                                                tokenizer=same_text)
    X = vectorizer.fit_transform(processed_tweets['text'])
    return vectorizer, X
#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        processed_tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
#     [YOUR CODE HERE]
    def changeLabel(label):
        if label in ['realDonaldTrump', 'mike_pence', 'GOP']:
            label = np.int32(0)
        else:
            label = np.int32(1)
        return label
    labels = processed_tweets['screen_name'].apply(changeLabel)
    return labels
#%%
class MajorityLabelClassifier():
    """
    A classifier that predicts the mode of training labels
    """
    def __init__(self):
        """
        Initialize your parameter here
        """
#         [YOUR CODE HERE]
        self.predicted_label = 0
        
    def fit(self, X, y):
        """
        Implement fit by taking training data X and their labels y and finding the mode of y
        i.e. store your learned parameter
        """
#         [YOUR CODE HERE]
        if np.count_nonzero(y==0) > np.count_nonzero(y==1):
            self.predicted_label = 0
        else:
            self.predicted_label = 1
    
    def predict(self, X):
        """
        Implement to give the mode of training labels as a prediction for each data instance in X
        return labels
        """
#         [YOUR CODE HERE]
        return [self.predicted_label for x in X]

#%%
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.SVC: classifier learnt from data
    """  
#     [YOUR CODE HERE]
    clf = sklearn.svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    return clf

#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
#     [YOUR CODE HERE]
    y_pred = classifier.predict(X_validation)
    return sklearn.metrics.accuracy_score(y_validation, y_pred)
#%%
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.SVC: classifier learned
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
#     [YOUR CODE HERE]
    processed_tweets = process_all(unlabeled_tweets)
    X = tfidf.transform(unlabeled_tweets['text'])
    return classifier.predict(X)