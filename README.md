# Deep_Learning_Final_Project

**Applied Deep Learning**

**Final Project Report**

**Sentiment analysis of reviews**

_Almas Aitken, BDA-2101_

_Github link:_ [_https://github.com/almasaitken/Deep\_Learning\_Final\_Project_](https://github.com/almasaitken/Deep_Learning_Final_Project)

_Video link:_ [_https://www.loom.com/share/15d298c577e749bebcedb7e08b8e2e71_](https://www.loom.com/share/15d298c577e749bebcedb7e08b8e2e71)

**Project overview** :

The goal of the project is to create a deep learning model which is able to classify movie reviews as either positive or negative. Since this is a NLP problem, recurrent neural networks are implemented in the model. The data used for training is the IMDB movie reviews dataset.

Different hyperparameters and models are implemented to identify the model with the highest accuracy. In the project, first, the same model is considered with different vocabulary sizes. Then the models with LSTM vs GRU layers are compared. Then the single direction recurrent layer vs bidirectional recurrent layer are compared. Then, the addition of a convolutional layer is considered.

**The dataset description:**

The IMDB movie reviews dataset includes 50,000 movie reviews which are evenly labeled either as positive or negative. In this project, the dataset from Keras is used. The dataset is already preprocessed. The words are converted to integers where the value of integer reveals the frequency of that word in the reviews. The lower the integer, the higher the frequency. The values 0, 1, 2 are reserved for padding "\<PAD\>", review start "\<START\>", and unknown words "\<UNK\>" correspondingly. The dataset can be loaded with different vocabulary sizes 'n' which means only 'n' most frequent words will be converted to frequency. Other words are converted to 'unknown'.

Example of a single tokenized review with padding applied:

![](RackMultipart20230219-1-oc4p6n_html_d9a6ac203602e280.png)

Example of word form (padding taken out):

"a short while in the \<UNK\> together they \<UNK\> upon a \<UNK\> place in the \<UNK\> that \<UNK\> an old \<UNK\> after \<UNK\> part of it they soon \<UNK\> its \<UNK\> \<UNK\> and \<UNK\> they may be able to use it to break through the \<UNK\> \<UNK\> br br black \<UNK\> is a very interesting \<UNK\> and i'm actually quite surprised that there aren't more films based on it as there's so much \<UNK\> for things to do with it it's \<UNK\> to say that \<UNK\> makes the best of it's \<UNK\> as despite it's \<UNK\> the film never actually feels \<UNK\> and \<UNK\> to \<UNK\> well throughout director \<UNK\> \<UNK\> \<UNK\> a great atmosphere for the film the fact that most of it takes place \<UNK\> the \<UNK\> \<UNK\> \<UNK\> \<UNK\> that the film feels very \<UNK\> and this \<UNK\>"

**The DL models description:**

First, the model with embedding, LSTM and dense layers is tested.

Embedding layer is the essential part of the model since it accepts the numerical input and maps it into a high-dimension vector. This high dimensionality allows one to determine the semantic meaning of each word, not only frequency. The input dimension is equal to the vocabulary size. The input length is equal to 300 and it is chosen as constant (REVIEW\_WORD\_COUNT). The output dimension 32 is chosen as it is not too big or small given the average review length of around 200.

LSTM layer is a bit advanced RNN which has a memory cell to remember more outputs from the previous iterations. This is important for capturing the context of each word.

Dense layer with sigmoid activation function is the final layer since we have a binary classification task for which this layer produces the probability of belonging to either negative or positive. Values closer to 0 signify negative sentiment and values closer to 1 signify positive sentiment.

![](RackMultipart20230219-1-oc4p6n_html_e16419bd4105adf8.png)

For the model above, the vocabulary size is changed to see if this is important for the model accuracy. The values tested are 1000, 3000 and 10000.

Then the model with GRU layer instead of LSTM is tested. GRU layer is more simple than LSTM so it can better fit simple models with fewer data than LSTM. GRU retains shorter memory than LSTM.

![](RackMultipart20230219-1-oc4p6n_html_5aa5c288cc2abf16.png)

Then the bidirectional GRU is implemented to see if it performs better than single direction. This might improve the accuracy since the meaning of the word can depend on both preceding and succeeding words.

![](RackMultipart20230219-1-oc4p6n_html_1e02ee63cb6ddc51.png)

Then the convolutional 1D layer is applied to see if it can improve the model. This might improve the accuracy since it can extract some lower level features of sentences. This convolutional layer is then tested against different vocabulary sizes so that it matches the complexity since more words are considered.

![](RackMultipart20230219-1-oc4p6n_html_40fcc7bf16a0da41.png)

As the final step, the dropout layer and L2 regularizers are added to prevent overfitting of the model.

![](RackMultipart20230219-1-oc4p6n_html_738c60f1f67a0ee8.png)

**Literature review with links (another solutions):**

- [https://www.kaggle.com/code/vincentman0403/sentimental-analysis-on-imdb-by-lstm](https://www.kaggle.com/code/vincentman0403/sentimental-analysis-on-imdb-by-lstm)

My solution is similar to this. This solution uses the IMDB reviews dataset from Keras which is already preprocessed. The model includes the LSTM layer. Finally, the model achieves the accuracy of 0.86.

- [https://github.com/hansmichaels/sentiment-analysis-IMDB-Review-using-LSTM/blob/master/sentiment\_analysis.py.ipynb](https://github.com/hansmichaels/sentiment-analysis-IMDB-Review-using-LSTM/blob/master/sentiment_analysis.py.ipynb)

This solution involves manual preprocessing of the IMDB dataset. The csv file contains two columns: review and label. Then the author strips the html tags, punctuation marks from the reviews and converts them to lowercase. The reviews are then tokenized using the Tokenizer library. The LSTM layer is used in the model. The author achieves an accuracy of 0.86.

- [https://www.tensorflow.org/tutorials/keras/text\_classification](https://www.tensorflow.org/tutorials/keras/text_classification)

This solution is similar to the two above, but it uses a TextVectorization layer to convert the text to numerical data. Accuracy of 0.87 is achieved.

- [https://medium.com/geekculture/sentiment-analysis-using-rnn-keras-e545fbe000](https://medium.com/geekculture/sentiment-analysis-using-rnn-keras-e545fbe000)

Similar to the first source. The author achieves 0.82 accuracy.

- [https://www.tensorflow.org/tutorials/keras/text\_classification\_with\_hub](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)

This solution is similar to the above, but has a bit different preprocessing approach. It involves using the pre-trained word embeddings (Google's NNLM). The accuracy achieved is 0.86.

**Results:**

First, the effect of the vocabulary size on the model accuracy is tested.

Below is the table showing the results:

| **Vocabulary size** | **Training accuracy** | **Validation Accuracy** | **Test accuracy** |
| --- | --- | --- | --- |
| 1000 | 0.8846 | 0.8444 | 0.8521 |
| 3000 | 0.9467 | 0.8602 | 0.8616 |
| 10000 | 0.9811 | 0.8512 | 0.8572 |

Graphs of the training accuracy vs validation accuracy:

_VOCAB\_SIZE=1000 VOCAB\_SIZE=3000_

![](RackMultipart20230219-1-oc4p6n_html_2c0a57fa25ce12c0.png) ![](RackMultipart20230219-1-oc4p6n_html_dc5826dd3d7379e8.png)

_VOCAB\_SIZE=10000_

![](RackMultipart20230219-1-oc4p6n_html_d8b66c86aa31514.png)

As can be seen from the above, the vocabulary size negatively affects the model since the overfitting increases as the vocabulary size increases.

Second, model performance with GRU layer is compared to that of the model with LSTM layer. Vocabulary size is 1000.

Below is the table showing the results:

| **Model** | **Training accuracy** | **Validation accuracy** | **Test accuracy** |
| --- | --- | --- | --- |
| GRU | 0.9124 | 0.8666 | 0.8700 |
| LSTM | 0.8846 | 0.8444 | 0.8521 |

Graphs of the training accuracy vs validation accuracy:

LSTM GRU

![](RackMultipart20230219-1-oc4p6n_html_2c0a57fa25ce12c0.png) ![](RackMultipart20230219-1-oc4p6n_html_4beeadb3324776c5.png)

As can be seen above, the model with GRU layer performs slightly better and produces almost the same overfitting rate. So, GRU can be considered as a better option.

Next, let's see if a bidirectional layer model improves the accuracy.

Below is the table showing the results:

| **Model** | **Training accuracy** | **Validation accuracy** | **Test accuracy** |
| --- | --- | --- | --- |
| Single direction GRU | 0.9124 | 0.8666 | 0.8700 |
| Bidirectional GRU | 0.9197 | 0.8664 | 0.8695 |

Graphs of the training accuracy vs validation accuracy:

Single Direction GRU Bidirectional GRU

![](RackMultipart20230219-1-oc4p6n_html_4beeadb3324776c5.png) ![](RackMultipart20230219-1-oc4p6n_html_6ae875f11ab3e2d8.png)

As can be seen above, the bidirectional layer almost makes no difference to either the accuracy and overfitting. So we keep the single direction model.

Then, we test the addition of a convolutional layer and compare it to the model without convolutional layer:

Below is the table showing the results:

| **Model** | **Training accuracy** | **Validation accuracy** | **Test accuracy** |
| --- | --- | --- | --- |
| Without CONV1D | 0.9124 | 0.8666 | 0.8700 |
| With CONV1D | 0.9517 | 0.8616 | 0.8623 |

Graphs of the training accuracy vs validation accuracy:

Without CONV1D layer With CONV1D layer

![](RackMultipart20230219-1-oc4p6n_html_4beeadb3324776c5.png) ![](RackMultipart20230219-1-oc4p6n_html_7299a7768d137041.png)

As can be seen above, the model with convolutional layer performs slightly worse and leads to greater overfitting, so no improvement.

Now let's change the vocabulary size to 10000 so to see if this better fits the model with a convolutional layer.

| **CONV1D** | **Training accuracy** | **Validation accuracy** | **Test accuracy** |
| --- | --- | --- | --- |
| Vocab\_size=1000 | 0.9517 | 0.8616 | 0.8623 |
| Vocab\_size=10000 | 0.9965 | 0.8704 | 0.8674 |

Graphs of the training accuracy vs validation accuracy:

CONV1D with VOCAB\_SIZE=1000 CONV1D with VOCAB\_SIZE=10000

![](RackMultipart20230219-1-oc4p6n_html_7299a7768d137041.png) ![](RackMultipart20230219-1-oc4p6n_html_71cf752628118b1b.png)

The accuracy improvement is insignificant, while there is even greater overfit for the data with greater vocabulary size.

**Discussion:**

First the model with single LSTM was tested against different vocabulary sizes. As a result, the bigger the vocabulary size, the bigger there is overfit. That suggests that the number of keywords which affect the sentiment is less. As the vocabulary size increases, the weights on the words which are almost insignificant are added which result in the overfit. Then the GRU layer model was compared against the LSTM layer model. GRU performed slightly better. Again, that is because the GRU layer fits more simple models and also because the average review size is not that big so GRU performs well. The bidirectional layer didn't perform better than single direction again due to a limited number of important words and they were located in different parts of the sentence so that single direction effect was canceled. Then the presence of a convolutional layer didn't suit the model for either with smaller and bigger vocabulary sizes. Again that is because of the model's simplicity.

To achieve greater accuracy, the dataset size could be increased so that more words and their corresponding sequences are taken into account. Other techniques such as data augmentation (synonym exchange), removing stop words, porter stemming could be applied to better preprocess the data. Overall, the accuracy of 87% is a pretty good value for classifying the reviews and it performs well on evident cases.

**Sources** :

- [https://www.kaggle.com/code/vincentman0403/sentimental-analysis-on-imdb-by-lstm](https://www.kaggle.com/code/vincentman0403/sentimental-analysis-on-imdb-by-lstm)
- [https://github.com/hansmichaels/sentiment-analysis-IMDB-Review-using-LSTM/blob/master/sentiment\_analysis.py.ipynb](https://github.com/hansmichaels/sentiment-analysis-IMDB-Review-using-LSTM/blob/master/sentiment_analysis.py.ipynb)
- [https://www.tensorflow.org/tutorials/keras/text\_classification](https://www.tensorflow.org/tutorials/keras/text_classification)
- [https://medium.com/geekculture/sentiment-analysis-using-rnn-keras-e545fbe000](https://medium.com/geekculture/sentiment-analysis-using-rnn-keras-e545fbe000)
- [https://www.tensorflow.org/tutorials/keras/text\_classification\_with\_hub](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)
- [https://keras.io/api/datasets/imdb/](https://keras.io/api/datasets/imdb/)
- [https://keras.io/api/layers/core\_layers/embedding/](https://keras.io/api/layers/core_layers/embedding/)
- [https://keras.io/api/layers/core\_layers/dense/](https://keras.io/api/layers/core_layers/dense/)
- [https://keras.io/api/layers/convolution\_layers/convolution1d/](https://keras.io/api/layers/convolution_layers/convolution1d/)
- [https://keras.io/api/layers/recurrent\_layers/gru/](https://keras.io/api/layers/recurrent_layers/gru/)
- [https://keras.io/api/layers/recurrent\_layers/lstm/](https://keras.io/api/layers/recurrent_layers/lstm/)
- [https://keras.io/api/layers/recurrent\_layers/bidirectional/](https://keras.io/api/layers/recurrent_layers/bidirectional/)
- [https://keras.io/api/layers/regularization\_layers/dropout/](https://keras.io/api/layers/regularization_layers/dropout/)
- [https://keras.io/api/layers/preprocessing\_layers/text/text\_vectorization/](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/)
