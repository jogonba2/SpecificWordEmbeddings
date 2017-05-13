import tensorflow as tf
import pickle as pck

class SpecificWordEmbeddings:
	
	def __init__(self, x, y, dim, window, n_hidden_cbow, n_hidden_spec, 
				 n_classes, mu, its, verbose=True, its_verbose=1):
		self.x = x
		self.y = y
		self.n_sents = len(x)
		self.cbow = not self.y
		self.dim = dim
		self.window = window
		self.n_hidden_cbow = n_hidden_cbow
		self.n_hidden_spec = n_hidden_spec
		self.n_classes = n_classes
		self.mu = mu
		self.its = its
		self.verbose = verbose
		self.its_verbose = 1
		self.vocab_size = -1
		self.trained = False
	
	def __get_contexts(self, x):
		contexts = []
		x = [0 for i in range(self.window)] + x
		x = x + [0 for j in range(self.window)]
		for i in range(self.window, len(x)-self.window): 
			contexts.append(x[i-self.window:i] + x[i:i+self.window+1])
		return contexts
	
	def get_vocab_size(self): return self.vocab_size
	
	def get_n_sents(self): return self.n_sents
	
	def get_cbow(self): return self.cbow
	
	def get_window(self): return self.window
	
	def get_n_hidden_cbow(self): return self.n_hidden_cbow
	
	def get_n_hidden_spec(self): return self.n_hidden_spec
	
	def get_n_classes(self): return self.n_classes
	
	def get_mu(self): return self.mu
	
	def get_its(self): return self.its
	
	def get_h(self): return self.j
	
	def __categorize(self):
		self.h  = {}
		self.vocab_size  = 0
		for i in range(len(self.x)):
			for j in range(len(self.x[i])):
				if self.x[i][j] not in self.h: 
					self.h[self.x[i][j]] = self.vocab_size
					self.vocab_size += 1
				self.x[i][j] = self.h[self.x[i][j]]
	
	def __set_simple_model(self):
		self.x_input = tf.placeholder("int32", self.window*2)
		self.y_ = tf.placeholder("int32", shape=(), name="Y_")
		self.W  = tf.Variable(tf.random_normal([self.vocab_size, self.dim]), name="W")
		W1 = tf.Variable(tf.random_normal([self.dim, self.n_hidden_cbow]))
		b1 = tf.Variable(tf.random_normal([self.n_hidden_cbow]))
		W2 = tf.Variable(tf.random_normal([self.n_hidden_cbow, self.dim]))
		b2 = tf.Variable(tf.random_normal([self.dim]))
		context_vectors = tf.nn.embedding_lookup(self.W, self.x_input)
		vec_i = tf.nn.embedding_lookup(self.W, self.y_)
		sum_node = tf.reduce_sum(context_vectors, 0)
		layer_1 = tf.nn.sigmoid(tf.matmul(tf.expand_dims(sum_node, 0), W1) + b1)
		layer_2 = tf.matmul(layer_1, W2) + b2
		self.loss = tf.reduce_mean(tf.square(tf.subtract(vec_i, layer_2)))
		self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
		self.init = tf.global_variables_initializer()
	
	def __set_specific_model(self):
		self.x_input = tf.placeholder("int32", self.window*2)
		self.y_ = tf.placeholder("int32", shape=(), name="Y_")
		self.c = tf.placeholder("float", self.n_classes)
		self.W  = tf.Variable(tf.random_normal([self.vocab_size, self.dim]), name="W")
		W1 = tf.Variable(tf.random_normal([self.dim, self.n_hidden_cbow]))
		b1 = tf.Variable(tf.random_normal([self.n_hidden_cbow]))
		W2 = tf.Variable(tf.random_normal([self.n_hidden_cbow, self.dim]))
		b2 = tf.Variable(tf.random_normal([self.dim]))
		W3 = tf.Variable(tf.random_normal([self.dim, self.n_hidden_spec]))
		b3 = tf.Variable(tf.random_normal([self.n_hidden_spec]))
		W4 = tf.Variable(tf.random_normal([self.n_hidden_spec, self.n_classes]))
		b4 = tf.Variable(tf.random_normal([self.n_classes]))
		context_vectors = tf.nn.embedding_lookup(self.W, self.x_input)
		vec_i = tf.nn.embedding_lookup(self.W, self.y_)
		sum_node = tf.reduce_sum(context_vectors, 0)
		layer_1 = tf.nn.sigmoid(tf.matmul(tf.expand_dims(sum_node, 0), W1) + b1)
		layer_2 = tf.matmul(layer_1, W2) + b2
		layer_3 = tf.nn.sigmoid(tf.matmul(tf.expand_dims(sum_node, 0), W3) + b3)
		layer_4 = tf.matmul(layer_3, W4) + b4
		self.loss = self.mu * tf.reduce_mean(tf.square(tf.subtract(vec_i, layer_2))) + \
					(1.0-self.mu) * tf.reduce_mean(tf.square(tf.subtract(self.c, layer_4)))
		self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
		self.init = tf.global_variables_initializer()

	def train(self):
		self.__categorize()
		if self.cbow: self.__set_simple_model() # No retraining
		else: self.__set_specific_model()
		self.sess = tf.Session()
		self.sess.run(self.init)
		for i in range(self.its):
			acum_loss = 0.0
			for j in range(len(self.x)):
				contexts = self.__get_contexts(self.x[j])
				if not self.cbow: truth = self.y[j]
				for context in contexts:
					y = context[self.window]
					context = context[0:self.window] + context[self.window+1:]
					if not self.cbow:
						self.sess.run(self.train_step, feed_dict={self.x_input:context, 
															 self.y_:y, self.c:truth})
						acum_loss += self.sess.run(self.loss, feed_dict={self.x_input:context, 
																	self.y_:y, self.c:truth})
					else:
						self.sess.run(self.train_step, feed_dict={self.x_input:context, self.y_:y})
						acum_loss += self.sess.run(self.loss, feed_dict={self.x_input:context, self.y_:y})
			if self.verbose and i%self.its_verbose==0:
				print("Epoch %d: %f" % (i, acum_loss/self.n_sents))
		if self.verbose: print("Model trained.")
		self.trained = True

	def get_embedding(self, word): 
		if not self.trained: return None
		id = self.h[word]
		embedding = tf.nn.embedding_lookup(self.W, self.y_)
		return self.sess.run(embedding, feed_dict={self.y_:id})
		
	def get_sent_embeddings(self, sent, dim, mode=0):
		res = []
		for i in range(len(sent)):
			embedding = self.get_embedding(sent[i])
			if embedding: res.append(embedding)
			else:
				if mode==1: res.append([0 for i in range(dim)])
		return res		

	def save(self, fname, tf_fname):
		if not self.trained: return -1
		obj = {"vocab_size": self.vocab_size, "n_sents": self.n_sents, 
			   "cbow": self.cbow, "dim": self.dim,
			   "window": self.window, "n_hidden_cbow": self.n_hidden_cbow,
			   "n_hidden_spec": self.n_hidden_spec, "n_classes": self.n_classes,
			   "mu": self.mu, "its": self.its, "h": self.h}
		with open(fname, "wb") as fw: pck.dump(obj, fw)
		saver = tf.train.Saver()
		saver.save(self.sess, tf_fname)
		
	def load(self, fname, tf_fname): 
		obj = {}
		with open(fname, "rb") as fr: obj = pck.load(fr)
		self.trained = True
		self.vocab_size = obj["vocab_size"]
		self.n_sents = obj["n_sents"]
		self.cbow = obj["cbow"]
		self.dim = obj["dim"]
		self.window = obj["window"]
		self.n_hidden_cbow = obj["n_hidden_cbow"]
		self.n_hidden_spec = obj["n_hidden_spec"]
		self.n_classes = obj["n_classes"]
		self.mu = obj["mu"]
		self.its = obj["its"]
		self.h = obj["h"]
		self.sess = tf.Session()
		loader = tf.train.import_meta_graph(tf_fname+".meta")
		loader.restore(self.sess, tf_fname)
		self.W  = tf.Variable(self.sess.run("W:0"))
		self.y_ = tf.placeholder("int32", shape=(), name="Y_")
		self.init = tf.global_variables_initializer()
		self.sess.run(self.init)
