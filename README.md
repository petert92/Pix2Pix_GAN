# Pix2Pix_GAN
Learning. Testing_Pix2Pix

Learning AI development.
From https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/ , Jason Brownlee.

Model:
"The GAN architecture is comprised of a generator model for outputting new plausible synthetic images and a discriminator model that classifies images as real ... 
The Pix2Pix model is a type of conditional GAN, or cGAN, where the generation of the output image is conditional on an input, in this case, a source image. The discriminator is provided both with a source image and the target image and must determine whether the target is a plausible transformation of the source image.
Again, the discriminator model is updated directly, and the generator model is updated via the discriminator model,..." How to Implement Pix2Pix GAN Models From Scratch With Keras, Jason Brownlee

Discriminator model is a PatchGAN, discriminator.py
Generator model is a U-Net, generator.py
GAN model = join D andG, join-G&D.py

Another post to train with Google Maps data set
https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/ , Jason Brownlee.

About the composite GAN to help understand it: "..This logical or composite model involves stacking the generator on top of the discriminator. A source image is provided as input to the generator and to the discriminator, although the output of the generator is connected to the discriminator as the corresponding “target” image. The discriminator then predicts the likelihood that the generator was a real translation of the source image.."

"... The number of epochs is set at 100 to keep training times down, although 200 was used in the paper. A batch size of 1 is used as is recommended in the paper.
Training involves a fixed number of training iterations. There are 1,097 images in the training dataset. One epoch is one iteration through this number of examples, with a batch size of one means 1,097 training steps. The generator is saved and evaluated every 10 epochs or every 10,970 training steps, and the model will run for 100 epochs, or a total of 109,700 training steps.
Each training step involves first selecting a batch of real examples, then using the generator to generate a batch of matching fake samples using the real source images. The discriminator is then updated with the batch of real images and then fake images.
Next, the generator model is updated providing the real source images as input and providing class labels of 1 (real) and the real target images as the expected outputs of the model required for calculating loss. The generator has two loss scores as well as the weighted sum score returned from the call to train_on_batch(). We are only interested in the weighted sum score (the first value returned) as it is used to update the model weights.
Finally, the loss for each update is reported to the console each training iteration and model performance is evaluated every 10 training epochs..."
