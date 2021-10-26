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
