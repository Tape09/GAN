import argparse
import cv2
import os
import glob
import GAN_models
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.optimizers import SGD,RMSprop

def load_image(path):
    #load one image
    img = cv2.imread(path, 1)
    # print(img.shape)
    # print(np.mean(img,axis=2))
    img = np.float32(img) / 127.5 - 1#zero centering the picture
    return img

def load_shuffle(paths,tail='*.png'):
    print("Loading images..")
    paths = glob.glob(os.path.join(paths, "*.png"))
    # Load images
    IMAGES = np.array([load_image(p) for p in paths])
    np.random.shuffle(IMAGES)
    print("Loading completed")
    return IMAGES

def generate_code():
    #latent code for generating pics
    z = np.random.uniform(-1, 1, 100)
    return z

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield i,l[i:i+n]
        # yield l[i:i+n]
def train(paths, batch_size, EPOCHS):

    #reproducibility
    #np.random.seed(42)

    fig = plt.figure()

    IMAGES = load_shuffle(paths)


    BATCHES = [ b for b in chunks(IMAGES, batch_size) ]

    discriminator = GAN_models.discriminator()
    generator = GAN_models.generator()
    discriminator_on_generator = GAN_models.generator_containing_discriminator(generator, discriminator)
    #try default params setting
    # adam_gen = Adam()
    # adam_dis = Adam()
    adam_gen=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    adam_dis=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_dis)

    print("Number of batches: {}".format(len(IMAGES)//batch_size))
    print("Batch size is {}".format(batch_size))

    #margin = 0.25
    #equilibrium = 0.6931
    inter_model_margin = 0.25

    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch))
        # load weights on first try (i.e. if process failed previously and we are attempting to recapture lost data)
        if epoch == 0:
            if os.path.exists('generator_weights') and os.path.exists('discriminator_weights'):
                print("Loading saves weights..")
                generator.load_weights('generator_weights')
                discriminator.load_weights('discriminator_weights')
                print("Finished loading")
            else:
                pass

        # generated_images = None#
        for index, image_batch in chunks(IMAGES,batch_size):
        # for index, image_batch in enumerate(BATCHES):
            print("Epoch {} Batch {}".format(epoch,index))

            Noise_batch = np.array( [ generate_code() for _ in range(len(image_batch)) ] )
            generated_images = generator.predict(Noise_batch)
            # for i, img in enumerate(generated_images):
            #     cv2.imwrite('results/{}.jpg'.format(i), np.uint8(255 * 0.5 * (img + 1.0)))

            Xd = np.concatenate((image_batch, generated_images))
            yd = [1] * len(image_batch) + [0] * len(image_batch) # labels

            # print("Training first discriminator..")
            d_loss = discriminator.train_on_batch(Xd, yd)

            Xg = Noise_batch
            yg = [1] * len(image_batch)

            # print("Training first generator..")
            g_loss = discriminator_on_generator.train_on_batch(Xg, yg)

            print("Generator loss: {} Discriminator loss: {} Total: {}".format(g_loss, d_loss, g_loss + d_loss))
            if g_loss < d_loss and abs(d_loss - g_loss) > inter_model_margin:#generator is better
                print("Updating discriminator..")
                while abs(d_loss - g_loss) > inter_model_margin:
                    # print("Updating discriminator..")
                    d_loss = discriminator.train_on_batch(Xd, yd)
                    print("Generator loss: {} Discriminator loss: {}".format(g_loss,d_loss))
                    if d_loss < g_loss:
                        break
            elif d_loss < g_loss and abs(d_loss - g_loss) > inter_model_margin:#discriminator is better
                print("Updating generator..")
                while abs(d_loss - g_loss) > inter_model_margin:
                    # print("Updating generator..")
                    g_loss = discriminator_on_generator.train_on_batch(Xg, yg)
                    print("Generator loss: {} Discriminator loss: {}".format(g_loss, d_loss))
                    if g_loss < d_loss:
                        break
            else:#D and G are about the same
                pass

            print("Final batch losses (after updates) : G", "Generator loss", g_loss, "Discriminator loss", d_loss, "Total:", g_loss + d_loss)

            if (index+1) % 20 == 0:
                print('Saving weights..')
                generator.save_weights('generator_weights', True)
                discriminator.save_weights('discriminator_weights', True)

        plt.clf()
        for i, img in enumerate(generated_images[:9]):
            i = i+1
            plt.subplot(3, 3, i)
            plt.imshow(img)
            plt.axis('off')
        fig.canvas.draw()
        plt.savefig('Epoch_' + str(epoch) + '.png')

def generate(img_num):
    '''
        Generate new images based on trained model.
    '''
    generator = GAN_models.generator()
    adam=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    generator.load_weights('generator_weights')

    noise = np.array( [ generate_code() for n in range(img_num) ] )

    print('Generating images..')
    generated_images = [img for img in generator.predict(noise)]
    for index, img in enumerate(generated_images):
        cv2.imwrite("{}.jpg".format(index), np.uint8(255 * 0.5 * (img + 1.0)))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str)
    parser.add_argument("--TYPE", type = str)
    parser.add_argument("--batch_size", type = int, default=50)
    parser.add_argument("--epochs", type = int, default = 2)
    #parser.add_argument("--handicap", type = int, default = 2)
    parser.add_argument("--img_num", type = int, default = 10)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # load_image('data128/0.png')
    # train('data128/',batch_size=64,EPOCHS=20)
    generate(2)

    # args = get_args()
    #
    # if args.TYPE == 'train':
    #     train(path = args.path, batch_size = args.batch_size, EPOCHS = args.epochs)
    #
    # elif args.TYPE == 'generate':
    #     generate(img_num = args.img_num)
