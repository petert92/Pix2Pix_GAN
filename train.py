from updateWeights import generate_real_samples, generate_fake_samples
from join_GandD import define_gan, define_generator, define_discriminator

""" one step train iteration"""

""" (1): select a batch of source and target images by calling generate_real_samples()
n_batch=1,  256x25 6assume image input => n_patch for D output=16x16 (according to actual design) """

""" (2) use the batches of selected real source images to generate corresponding batches of generated or fake target images"""

""" (3) use the real and fake images, as well as their targets, to update the standalone discriminator model"""


""" (4) Again, the discriminator model is updated directly, and the generator model is updated via the discriminator model, although the loss function is updated. 
The generator is trained via adversarial loss, which encourages the generator to generate plausible images in the target domain. 
The generator is also updated via L1 loss measured between the generated image and the expected output image. 
This additional loss encourages the generator model to create plausible translations of the source image."""
""" (4) two loss functions, but three loss values calculated for a batch update, only interest on the weighted sum of the adversarial and L1 loss values for the batch."""

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1, n_patch=16):
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples (1)
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples (2)
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples (3)
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples (3)
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator (4)
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('&gt;%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

...
# load image data
dataset = ...
# define image shape
image_shape = (256,256,3)
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)