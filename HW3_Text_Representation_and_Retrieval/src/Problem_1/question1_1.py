import nltk
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import argparse
from operator import itemgetter
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")

parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")
parser.add_argument('--n_epochs', type=int, default=5, help="number of epochs for training")
parser.add_argument('--context_size', type=int, default=2, help="window size for context")
parser.add_argument('--embedding_size', type=int, default=64, help="size for each word embedding")
parser.add_argument('--learning_rate', type=float, default=1e-5, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.5, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

FLAGS = parser.parse_args()

class SkipGramModel(nn.Module):
	def __init__(self, vocab_size, embedding_size):
		super(SkipGramModel, self).__init__()
		self.linear1 = nn.Linear(vocab_size, embedding_size)
		self.linear2 = nn.Linear(embedding_size, vocab_size)
		self.softmax = nn.Softmax()
		
	def forward(self, context):
		cxt_emb = self.linear1(context.view(FLAGS.batch_size, -1))
		prob = F.log_softmax(self.linear2(cxt_emb), dim=1)

		return prob

def vectorize(index):
	vector = torch.zeros(vocab_size)
	vector[index] = 1
	return vector

def vectorize_batches(index):
	vector = torch.zeros(FLAGS.batch_size, vocab_size)
	for i in range(index.shape[0]):
		vector[:, index[i]] = 1
	return vector

def vectorize_test_batches(index):
	vector = torch.zeros(1, vocab_size)
	for i in range(index.shape[0]):
		vector[:, index[i]] = 1
	return vector

# setting the parameters
context_size = FLAGS.context_size
embedding_size = FLAGS.embedding_size

# To be downloaded for the first time
#nltk.download('abc')
text_corpus = nltk.corpus.abc.raw()
word_splits = text_corpus.split()
word_splits = list(map(lambda x: x.rstrip(), word_splits))

# train_corpus = word_splits[:(int)(0.8*(len(word_splits)))]
# test_corpus = word_splits[(int)(0.8*(len(text_corpus))):]

vocabulary = set(word_splits)
vocab_size = len(vocabulary)
print('vocabulary created: ', len(vocabulary), ' words')


word_to_idx = {}
idx_to_word = {}

for i, word in enumerate(vocabulary):
	word_to_idx[word] = i
	idx_to_word[i] = word

# using the skip-gram model --> creating data
skip_gram_data = []
for i in range(context_size, len(word_splits)-context_size-1):
	
	current_word = word_splits[i]

	for w in range(-context_size, context_size+1):
		if(w!=0):
			context_window_word = word_splits[i+w]
			skip_gram_data.append((current_word, context_window_word, 1)) # appending positive samples

	# for appending negative samples
	# for j in range(-context_size, context_size+1):
	# 	r = random.randint(0, 1)
	# 	if(r==0 or i>=len(word_splits)-3):
	# 		r1 = random.randint(0, i-1)
	# 	else:
	# 		r1 = random.randint(i+3, len(word_splits)-1)
	# 	skip_gram_data.append((current_word, word_splits[r1], 0))

loss = nn.NLLLoss()
model = SkipGramModel(vocab_size, embedding_size)
optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
optimizer_adam = optim.Adam(list(model.parameters()), lr=FLAGS.learning_rate, betas=(FLAGS.beta_1, FLAGS.beta_2))
word_losses = []

vocab_list = list(vocabulary)
key_words = []
n = 10
top_n = 15

for i in range(n):
	index = random.randint(0, vocab_size)
	key_words.append(vocab_list[index])

if(FLAGS.cuda):
	loss = loss.cuda()
	model = model.cuda()

def word_to_emb(word, model):
	tgt_idx = word_to_idx[word]
	tgt_idx = Variable(torch.LongTensor([tgt_idx])).cuda()
	tgt_idx = vectorize_test_batches(tgt_idx).cuda()
	return model.linear1(tgt_idx).cpu().data.numpy()

def train_in_batches():
	for i in range(FLAGS.n_epochs):
		optimizer_adam.zero_grad()
		
		total_loss = 0.0
		
		for batch_idx in range(len(skip_gram_data) // FLAGS.batch_size):
			batch_data = skip_gram_data[batch_idx*FLAGS.batch_size: (batch_idx+1)*FLAGS.batch_size]
			batch_data = np.asarray(batch_data)
			
			if(batch_data.shape[0]!=FLAGS.batch_size):
				continue
			tgt, cxt, value = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
			
			tgt_idx = itemgetter(*list(tgt))(word_to_idx)
			tgt_idx = Variable(torch.LongTensor(list(tgt_idx))).cuda()

			cxt_idx = itemgetter(*list(cxt))(word_to_idx)
			cxt_idx = Variable(torch.LongTensor(list(cxt_idx))).cuda()

			cxt_vector = vectorize_batches(cxt_idx).cuda()
			# value = Variable(torch.from_numpy(value.astype('float32')))

			model_output = model(cxt_vector)
			loss_val = loss(model_output, tgt_idx)

			loss_val.backward()
			optimizer_adam.step()

			total_loss += loss_val.item()
			
			if(batch_idx%5000==0):
				print('Iteration: ', batch_idx, ' Total Loss: ', total_loss / (batch_idx * FLAGS.batch_size + 1))
		
		word_losses.append(total_loss)
		print('Epoch: ', i, word_losses[i] / len(skip_gram_data))
		
		torch.save(model.state_dict(), './model_'+ str(i) + '.pth')
		visualisation(model, filename='visualisation_epoch_'+(str)(i))

	return model, word_losses


def train():
	for i in range(FLAGS.n_epochs):
		model.zero_grad()
		total_loss = 0.0
		counter = 0

		for tgt, cxt, value in skip_gram_data:

			tgt_idx = Variable(torch.LongTensor([word_to_idx[tgt]]))
			cxt_idx = Variable(torch.LongTensor([word_to_idx[cxt]]))
			value = Variable(torch.Tensor([value]))
			
			cxt_vector = vectorize(cxt_idx)
			model_output = model(cxt_vector)
			loss_val = loss(model_output.view(1, -1), tgt_idx)
			loss_val.backward()
			optimizer.step()

			total_loss += loss_val.item()
			if(counter%5000==0):
				print(total_loss, '=========', counter)
			counter += 1
		word_losses.append(total_loss)
		print('Epoch: ', i, word_losses[i])

	return model, word_losses

def visualisation(model, filename):
	# randomly selecting n words
	n = 10
	top_n = 15

	vocab_embeddings = np.zeros((vocab_size, embedding_size))
	key_word_embeddings = np.zeros((n, embedding_size))

	for k in range(len(key_words)):
		key = key_words[k]
		key_emb = word_to_emb(key, model)
		key_word_embeddings[k, :] = key_emb

	for w in range(len(vocab_list)):
		word = vocab_list[w]
		word_emb = word_to_emb(word, model)
		vocab_embeddings[w, :] = word_emb

	sim_matrix = cosine_similarity(key_word_embeddings, vocab_embeddings)

	embedding_clusters = []
	word_clusters = []
	
	for idx, word in enumerate(key_words):
		embeddings = []
		words = []

		similar_words = np.argsort(sim_matrix[idx, :])[-top_n:]
		
		for similar_word in similar_words:
			words.append(vocab_list[similar_word])
			embeddings.append(vocab_embeddings[similar_word, :])
		embedding_clusters.append(embeddings)
		word_clusters.append(words)

	embedding_clusters = np.array(embedding_clusters)
	n, m, k = embedding_clusters.shape
	tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
	embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

	tsne_plot_similar_words('Similar words from abc corpus', key_words, embeddings_en_2d, word_clusters, 0.7, filename)

def visualise_entire_vocab(model, epoch):
	vocab_embeddings = np.zeros((vocab_size, embedding_size))

	for w in range(len(vocab_list)):
		word = vocab_list[w]
		word_emb = word_to_emb(word, model)
		vocab_embeddings[w, :] = word_emb

	vocab_embeddings = np.array(vocab_embeddings)
	X = TSNE(n_components=2, perplexity=100).fit_transform(vocab_embeddings)

	vis_x = X[:, 0]
	vis_y = X[:, 1]

	fig, ax = plt.subplots(1)
	ax.set_yticklabels([])
	ax.set_xticklabels([])

	plt.scatter(vis_x, vis_y, marker='.', c='r', cmap=plt.cm.get_cmap("jet", 10), s=3)
	plt.axis('off')
	plt.clim(-0.5, 10.5)

	plt.savefig('tsne_visualization_epoch_'+str(epoch)+'.png')

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')

### FOR TRAINING
# model, losses = train_in_batches()
# torch.save(model.state_dict(), './model_last_epoch.pth')

#### FOR EVALUATION
model = SkipGramModel(vocab_size, embedding_size).cuda()

for epoch in range(1, FLAGS.n_epochs):
	model.load_state_dict(torch.load('./model_'+str(epoch)+'.pth'))
	print('Loaded Model')
	visualise_entire_vocab(model, epoch)