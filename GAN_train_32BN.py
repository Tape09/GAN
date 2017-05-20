import argparse
import cv2
import os
import glob
import GAN_models_32BN as GAN_models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.optimizers import Adam
from keras.optimizers import SGD,RMSprop
from pprint import pprint
from tqdm import tqdm
from PIL import Image

def load_image(path):
    #load one image
    # img = cv2.imread(path, 1)


    # img = Image.open(path)
    # img = np.array(img.getdata(),dtype=float).reshape(img.size[0], img.size[1], 3)

    img = mpimg.imread(path)
    # plt.imshow(img)
    # plt.show()
    # # print(img.shape)
    # # print(np.mean(img,axis=2))
    # print(type(img))
    img = np.float32((img/ 127.5) - 1)#zero centering the picture
    return img

def load_shuffle(paths,tail='*.png'):
    print("Loading images..")
    paths = glob.glob(os.path.join(paths, tail))
    # Load images
    IMAGES = np.array([load_image(p) for p in paths])
    np.random.shuffle(IMAGES)
    print("Loading completed")
    return IMAGES

def generate_code(n=120):
    #120 breeds
    #latent code for generating pics
    z = np.random.uniform(-1, 1, n)
    # z = np.random.normal(0, 1, n)
    return z

def generate_code_batch(batch_size):
    return np.array([generate_code() for _ in range(batch_size)])

def train(paths, batch_size, EPOCHS):

    IMAGES = load_shuffle(paths)

    discriminator = GAN_models.discriminator()
    generator = GAN_models.generator()
    discriminator_on_generator = GAN_models.generator_containing_discriminator(generator, discriminator)

    # sgd_gen = SGD(lr=0.00002, decay=0, momentum=0.5, nesterov=True)
    # sgd_dis = SGD(lr=0.00002, decay=0, momentum=0.5, nesterov=True)
    # generator.compile(loss='binary_crossentropy', optimizer=sgd_gen)
    # discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=sgd_gen)
    # discriminator.trainable = True
    # discriminator.compile(loss='binary_crossentropy', optimizer=sgd_dis)

    adam_gen=Adam(lr=0.0001)
    adam_dis=Adam(lr=0.001)
    generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_dis)

    # rmsprop_gen = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0)
    # rmsprop_dis = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0)
    # generator.compile(loss='binary_crossentropy', optimizer=rmsprop_gen)
    # discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=rmsprop_gen)
    # discriminator.trainable = True
    # discriminator.compile(loss='binary_crossentropy', optimizer=rmsprop_dis)

    print("Batch size is {}".format(batch_size))
    dLosses = []
    gLosses = []
    k = 1
    image_val = 0.9
    for epoch in tqdm(range(EPOCHS)):
        # print("Epoch {}".format(epoch))
        # load weights on first try (i.e. if process failed previously and we are attempting to recapture lost data)
        if epoch == 0:
            if os.path.exists('generator_weights_32BN') and os.path.exists('discriminator_weights_32BN'):
                print("Loading saves weights..")
                generator.load_weights('generator_weights_32BN')
                discriminator.load_weights('discriminator_weights_32BN')
                print("Finished loading")
            else:
                pass


        d_loss =0
        for i in range(k):
            image_batch = IMAGES[np.random.randint(0, IMAGES.shape[0], size=batch_size)]
            Noise_batch = generate_code_batch(batch_size)
            generated_images = generator.predict(Noise_batch)
            # for i, img in enumerate(generated_imkages):
            #     cv2.imwrite('results/{}.jpg'.format(i), np.uint8(255 * 0.5 * (img + 1.0)))
            Xd = np.concatenate((image_batch, generated_images))
            yd = [image_val] * batch_size + [0] * batch_size # labels
            d_loss += discriminator.train_on_batch(Xd, yd)
        d_loss /=k
        dLosses.append(d_loss)

        Xg = generate_code_batch(batch_size)
        yg = [image_val] * batch_size

        g_loss = discriminator_on_generator.train_on_batch(Xg, yg)
        gLosses.append(g_loss)

        # print("D loss: {} G loss: {}".format(d_loss,g_loss))
        if (epoch+1)%20000==0:
            print('Epoch {} ,Saving weights..'.format(epoch))
            generator.save_weights('generator_weights_32BN', True)
            discriminator.save_weights('discriminator_weights_32BN', True)
        if (epoch + 1) % 2000 == 0:
            plot_generated_images(generator=generator,epoch=epoch)


    plot_loss(dLosses,gLosses)

def generate(img_num):
    '''
        Generate new images based on trained model.
    '''
    generator = GAN_models.generator()

    # adam=Adam(lr=0.00005, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    # generator.compile(loss='binary_crossentropy', optimizer=adam)

    sgd_gen = SGD(lr=0.0002, decay=0, momentum=0.5, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer=sgd_gen)

    # rmsprop = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0)
    # generator.compile(loss='binary_crossentropy', optimizer=rmsprop)

    generator.load_weights('generator_weights_64BN')

    noise = np.array( [ generate_code() for _ in range(img_num) ] )

    print('Generating images..')
    generated_images = [img for img in generator.predict(noise)]
    for index, img in enumerate(generated_images):
        cv2.imwrite("{}.jpg".format(index), np.uint8(255 * 0.5 * (img + 1.0)))

def plot_loss(d,g):
    plt.figure(figsize=(10,8))
    plt.plot(d,label = "discrminitive loss")
    plt.plot(g,label = 'generative loss')
    plt.legend()
    plt.show()

def plot_generated_images(generator,epoch,path ='result_32BN'):
    # print("saving snapshoot")
    fig = plt.figure()
    Noise_batch = generate_code_batch(9)
    generated_images = generator.predict(Noise_batch)
    print(generated_images.shape)
    plt.clf()
    for i, img in enumerate(generated_images[:9]):
        i = i + 1
        plt.subplot(3, 3, i)
        img = np.uint8(255 * 0.5 * (img + 1.0))
        plt.imshow(img)
        plt.axis('off')
    fig.canvas.draw()
    path += '/32_Epoch_' + str(epoch) + '.png'
    plt.savefig(path)

if __name__ == "__main__":
    # TODO:1)use non-saturating game
    # TODO:2)use label smoothing
    # TODO:3)do not apply sigmoid at the output of D
    # TODO:4)use RMSProp instead of ADAM
    # TODO:5)lower learning rate,e.g eta=0.00005
    # TODO:6)include auxiliary information
    # TODO: 7)use reference BN instead of normal BN, cuz normal BN will introduce intra samples correlation
    # TODO: 8) defining the fenerator objective with respect to an unrolled optimization of D
    # load_image('data128/0.png')
    train('test32/',batch_size=128,EPOCHS=40000)
    # generate(2)