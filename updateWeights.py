from numpy import randint, ones, zeros

"""helper function that will select a batch of real source and target images and the associated output (1.0)"""
## select a batch of random samples, returns images and target
# input: dataset, real samples total,  output matrix shape
# output: 2 arrays of random samples, ones matrix(output)
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

"""function to generate a batch of fake images and the associated output (0.0)"""
## generate a batch of images, returns images and targets
# input: generator model, samples total, ones matrix(output)
# output: array of fake samples, zeros matrix(output)
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y