# English-Dutch-Classifier_DecisionTree
This python application differentiates between English and Dutch sentences and provides a decision based on training data for every sentence in the file provided.
# Input Processing
1. The Training data is taken from the file from the filename provided in the command line arguments and processed.
2. The features are created which will be categorizing the data as “Present” and “Not Present”
3. Following are the features used. The main focused on the selection of features is on the main part of the language; Grammar and Parts of Speech.
a. f1_article
b. f2_pronoun
c. f3_average_length_of_word = 4.7
d. f4_wh_words
e. f5_simple_preposition
f. f6_double_preposition
g. f7_past_tense
h. f8_future_tense
i. f9_negation
j. f10_dutch_words
4. The whole dataset is processed through every feature described below. It is then converted into every entry in the dataset being binary i.e. “Present” and “Not Present”.
5. This converted dataset is then passed through the decision tree algorithm.
# Decision Tree
1. The decision tree consists of a tree node with the following values.
a. Value
b. Present
c. Not_Present
2. The Present and Not Present are the branches of the node. After the dataset is passed through the root node, the maximum information gain of a feature amongst all the features is calculated through the function and that feature is expanded. The root’s value is the feature and the present and not present branches are the dataset where the feature’s value in the dataset was present and not present respectively.
3. After selecting the algorithm, we then use recursion to calculate the whole tree where the base condition to break the recursion is root’s value to be either “en” or “nl”, which are the classification results and hence, the leaf node
4. After a particular feature is selected, the column of that feature is removed from the dataset and everything is calculated again for the remaining nodes.
5. The trained node is stored in a file as an object using serialization.
6. For predicting the data, we load the trained object from the file and again process the dataset through all the features and converting the data into values of “Present” and “Not Present”.
7. We now process the dataset line by line and pass through the trained object. The evaluation is the class of the language. i.e. “en” or “nl”
8. The prediction accuracy of the algorithm is 90% for the dataset provided by the instructor.
# How to run the application
train <examples> <hypothesisOut> <learning-type> should read in labeled examples and perform some sort of training.
examples is a file containing labeled examples.
hypothesisOut specifies the file name to write your model to.
learning-type specifies the type of learning algorithm you will run, it is "dt"
predict <hypothesis> <file> Your program should classify each line as either English or Dutch using the specified model. Note that this must not do any training, but should take a model and make a prediction about the input. For each input example, your program should simply print its predicted label on a newline. For example. It should not print anything else.
hypothesis is a trained decision tree or ensemble created by your train program
file is a file containing lines of 15 word sentence fragments in either English or Dutch
