# Text Sentiment Classification

The "run.py" allow the user to classify the Sentiment of tweet in a binary manner, namely positive or negative.
The programme creates a file "kaggle_submission.csv" that correspond to the submission selected on kaggle.

## Installation
The machine that will run the programs present in this repo should have installed:<br>
		- python3 https://www.python.org/downloads/ <br>
		- the numpy library http://www.numpy.org/<br>
		- the nlkt library http://www.nltk.org/<br>
		- the sklearn library http://scikit-learn.org/<br>
		- the matplotlib library https://matplotlib.org<br>
		- the tensorflow library https://www.tensorflow.org<br>
		- the keras library https://keras.io/<br>

The training and test data should be downloaded from kaggle and unziped in the "Proj_2" folder:<br>
	- download link https://www.kaggle.com/c/epfml17-text/data

For convenience of the user the "run.py" uses an already trained and saved model. This model is an MLP Classifier and is availabel here **link**. 
Download the pickle file "Optimal_mlpc.pkl" and save it inside the "Proj_2" folder.

The run.py uses the sent2vec algorithm and uses a pre-trained model. In order to be able to run the programm you should:<br>
	- download the sent2vec_twitter_bigrams and save it inside the "Proj_2" folder. Be aware that this file is large (23GB)<br>
	link: https://drive.google.com/open?id=0B6VhzidiLvjSeHI4cmdQdXpTRHc<br>
	- run the `make command inside the "Proj_2" folder to compile the Facebook's FastText library that is used by the sent2vec algorithm.


## Running the code

Run the following command in your terminal inside the folder "Proj_2":

$python3 run.py

The "run.py" programm will embedd the "test_data.csv" and use the "Optimal_mlpc" trained model to output a prediction.


## Other Runable Programs

We provide 3 different runable code that doesn't use saved model and that create them. We creates such programs for three different methods.

### run_base.py
This program produced the submission of our baseline. It uses an external dataset for the embedding. The embedding was trained with the word2vec algorithm. The sentence embedding is represented as the average of the embedding of the words that were present in the vocabulary.

In order to be able to run "run_base.py" you should download the embedding dataset:<br>
	- Click on this link to download the word embedding http://4530.hostserv.eu/resources/embed_tweets_en_200M_200D.zip<br>
	- Unzip the file in the "Proj_2" folder<br>
	- run the `$python3 run_base.py`<br>

This programme will embedd each tweet using the average of all word embedding and then use an MLP classifier to make the prediction.

### run_RNN.py
This programm is an improvement of the baseline. It uses the same word embedding but feed thos embedding to a recurrent neural network instead of averaging word.

In order to be able to run "run_RNN.py" you should download the embedding dataset:<br>
	- Click on this link to download the word embedding http://4530.hostserv.eu/resources/embed_tweets_en_200M_200D.zip<br>
	- Unzip the file in the "Proj_2" folder<br>
	- run the `$python3 run_RNN.py`<br>

### run_s2v.py
This programm is similar to "run.py" but doesn't use a pre-saved model.
In order to be able to run "run_s2v.py" you should download the twitter bigram

## Folder Organisation

### - embedding.py
	Contains the methods used to produce the embeddings of the tweets whether is based on word or sentence embedding:
		avg_words(words)
		sentence_embedding(train_pos_filename, train_neg_filename, test_data_filename)
		word_embedding(embedding_filename, vocab_filename)
		sentence_avg_representation(train_pos, train_neg, test_data, embedding)
### - training.py
	Contains the methods used to construct the differents trained models:
		mlpc_model_for_s2v(train_data, train_labels, nb_neur, alpha, depth, save=False)
		rnn_model_for_w2v()
		mlpc_model_for_w2v(train_data, train_labels, test_data)
		validation_s2v(train_data, train_labels, alphas, neurones, depth)
### - hw_helpers.py
	Contains some helpers methods that were implemented during our lab sessions:
		create_csv_submission(ids, y_pred, name)
		plot_train_test(train_errors, test_errors, alphas, nb_neurones)
### - sent2vec.py
	Contains the code that generate the embedding of sentences, this code was tooken from the notebook provided by the author of this algorithm. 
	Reference: Matteo Pagliardini, Prakhar Gupta, Martin Jaggi, [*Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features*](https://arxiv.org/abs/1703.02507)
		tokenize(tknzr, sentence, to_lower=True)
		format_token(token)
		tokenize_sentences(tknzr, sentences, to_lower=True)
		get_embeddings_for_preprocessed_sentences(sentences, model_path, fasttext_exec_path)
		read_embeddings(embeddings_path)
		dump_text_to_disk(file_path, X, Y=None)
		get_sentence_embeddings(sentences)
### - validation.py
	The code that was used to produced the train/validation error plots
	Contains a main function:
			validate()

### - run_s2v.py
	The full implementation of the Text sentiment classification using sent2vec.
	Contains a main function:
    	run()
### - run_RNN.py
	The full implementation of the Text sentiment classification using word embedding and a recurent neural network.
	Contains a main funtion:
		run()
### - run_base.py
	The full implementation of the Text Sentiment Classification using word embedding, average representation of tweets and MLP classifier
	Contains a main function:
		run()


## Additional notes

For complementary informations about the methods check the DocStrings.
And for comprehensive informations about the detailed thinking behind this code check the project report

