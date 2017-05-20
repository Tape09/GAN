# import argparse
# import cv2
# import os
# import glob
# import GAN_models_64 as GAN_models
# import matplotlib.pyplot as plt
# import numpy as np
# from keras.optimizers import Adam
# from keras.optimizers import SGD,RMSprop
#
# # global lr = 0.00001
#
# def load_image(path):
#     #load one image
#     img = cv2.imread(path, 1)
#     # print(img.shape)
#     # print(np.mean(img,axis=2))
#     img = np.float32(img) / 127.5 - 1#zero centering the picture
#     return img
#
# def load_shuffle(paths,tail='*.png'):
#     print("Loading images..")
#     paths = glob.glob(os.path.join(paths, "*.png"))
#     # Load images
#     IMAGES = np.array([load_image(p) for p in paths])
#     np.random.shuffle(IMAGES)
#     print("Loading completed")
#     return IMAGES
#
# def generate_code(n=100):
#     #latent code for generating pics
#     z = np.random.uniform(-1, 1, n)
#     return z
#
# def generate_code_batch(batch_size):
#     return np.array([generate_code() for _ in range(batch_size)])
#
# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield i,l[i:i+n]
#         # yield l[i:i+n]
#
# def train(paths, batch_size, EPOCHS):
#     #reproducibility
#     #np.random.seed(42)
#
#     fig = plt.figure()
#     IMAGES = load_shuffle(paths)
#     # BATCHES = [ b for b in chunks(IMAGES, batch_size) ]
#
#
#     discriminator = GAN_models.discriminator()
#     generator = GAN_models.generator()
#     discriminator_on_generator = GAN_models.generator_containing_discriminator(generator, discriminator)
#
#     sgd_gen = SGD(lr=0.0002, decay=0, momentum=0.5, nesterov=True)
#     sgd_dis = SGD(lr=0.0002, decay=0, momentum=0.5, nesterov=True)
#     generator.compile(loss='binary_crossentropy', optimizer=sgd_gen)
#     discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=sgd_gen)
#     discriminator.trainable = True
#     discriminator.compile(loss='binary_crossentropy', optimizer=sgd_dis)
#
#
#
#
#
#     # adam_gen=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
#     # adam_dis=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
#     # generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
#     # discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=adam_gen)
#     # discriminator.trainable = True
#     # discriminator.compile(loss='binary_crossentropy', optimizer=adam_dis)
#
#     # rmsprop_gen = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0)
#     # rmsprop_dis = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0)
#     # generator.compile(loss='binary_crossentropy', optimizer=rmsprop_gen)
#     # discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=rmsprop_gen)
#     # discriminator.trainable = True
#     # discriminator.compile(loss='binary_crossentropy', optimizer=rmsprop_dis)
#
#     print("Number of batches: {}".format(len(IMAGES)//batch_size))
#     print("Batch size is {}".format(batch_size))
#
#     #margin = 0.25
#     #equilibrium = 0.6931
#     inter_model_margin = 0.1
#
#     for epoch in range(EPOCHS):
#         print("Epoch {}".format(epoch))
#         # load weights on first try (i.e. if process failed previously and we are attempting to recapture lost data)
#         if epoch == 0:
#             # print(os.path.exists('generator_weights_64'), os.path.exists('discriminator_weights_64'))
#             if os.path.exists('generator_weights_64') and os.path.exists('discriminator_weights_64'):
#                 print("Loading saves weights..")
#                 generator.load_weights('generator_weights_64')
#                 discriminator.load_weights('discriminator_weights_64')
#                 print("Finished loading")
#             else:
#                 pass
#         d,g,p=0,0,0
#         # generated_images = None#
#         for index, image_batch in chunks(IMAGES,batch_size):
#         # for index, image_batch in enumerate(BATCHES):
#             print("Epoch {} Batch {}".format(epoch,index))
#
#             # Noise_batch = np.array( [ generate_code() for _ in range(len(image_batch)) ] )
#             Noise_batch = generate_code_batch(len(image_batch))
#             generated_images = generator.predict(Noise_batch)
#             # for i, img in enumerate(generated_images):
#             #     cv2.imwrite('results/{}.jpg'.format(i), np.uint8(255 * 0.5 * (img + 1.0)))
#             image_val = 0.9
#
#             Xd = np.concatenate((image_batch, generated_images))
#             yd = [image_val] * len(image_batch) + [0] * len(image_batch) # labels
#
#             # print("Training first discriminator..")
#             d_loss = discriminator.train_on_batch(Xd, yd)
#
#             Xg = Noise_batch
#             yg = [image_val] * len(image_batch)
#
#             # print("Training first generator..")
#             g_loss = discriminator_on_generator.train_on_batch(Xg, yg)
#
#             # print("Generator loss: {} Discriminator loss: {} Total: {}".format(g_loss, d_loss, g_loss + d_loss))
#             if g_loss < d_loss and abs(d_loss - g_loss) > inter_model_margin:#generator is better
#                 # print("Updating discriminator..")
#                 d+=1
#                 while abs(d_loss - g_loss) > inter_model_margin:
#                     # print("Updating discriminator..")
#                     d_loss = discriminator.train_on_batch(Xd, yd)
#                     # print("\rGenerator loss: {} Discriminator loss: {}".format(g_loss,d_loss),end='\r')
#                     if d_loss < g_loss:
#                         break
#             elif d_loss < g_loss and abs(d_loss - g_loss) > inter_model_margin:#discriminator is better
#                 # print("Updating generator..")
#                 g+=1
#                 while abs(d_loss - g_loss) > inter_model_margin:
#                     # print("Updating generator..")
#                     g_loss_old = g_loss
#                     g_loss = discriminator_on_generator.train_on_batch(Xg, yg)
#                     if abs(g_loss-g_loss_old)<1e-8:
#                         print("Are you really learning?")
#
#                     # print("\rGenerator loss: {} Discriminator loss: {}".format(g_loss, d_loss),end='\r')
#                     if g_loss < d_loss:
#                         break
#             else:#D and G are about the same
#                 p+=1
#                 pass
#
#             # print("Final batch losses (after updates) : G", "Generator loss", g_loss, "Discriminator loss", d_loss, "Total:", g_loss + d_loss)
#             print("D: {} G: {} P: {}".format(d,g,p))
#
#         print('Epoch {} ,Saving weights..'.format(epoch))
#         generator.save_weights('generator_weights_64', True)
#         discriminator.save_weights('discriminator_weights_64', True)
#
#         plt.clf()
#         for i, img in enumerate(generated_images[:9]):
#             i = i+1
#             plt.subplot(3, 3, i)
#             img =np.uint8(255 * 0.5 * (img + 1.0))
#             plt.imshow(img)
#             plt.axis('off')
#         fig.canvas.draw()
#         plt.savefig('result_64/64_Epoch_' + str(epoch+161) + '.png')
#
# def generate(img_num):
#     '''
#         Generate new images based on trained model.
#     '''
#     generator = GAN_models.generator()
#
#     # adam=Adam(lr=0.00002, beta_1=0.0005, beta_2=0.999, epsilon=1e-08)
#     # generator.compile(loss='binary_crossentropy', optimizer=adam)
#
#     sgd_gen = SGD(lr=0.0002, decay=0, momentum=0.5, nesterov=True)
#     generator.compileile(loss='binary_crossentropy', optimizer=sgd_gen)
#
#     # rmsprop = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0)
#     # generator.compile(loss='binary_crossentropy', optimizer=rmsprop)
#
#     generator.load_weights('generator_weights_64')
#
#     noise = np.array( [ generate_code() for _ in range(img_num) ] )
#
#     print('Generating images..')
#     generated_images = [img for img in generator.predict(noise)]
#     for index, img in enumerate(generated_images):
#         cv2.imwrite("results/{}.jpg".format(index), np.uint8(255 * 0.5 * (img + 1.0)))
#
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", type = str)
#     parser.add_argument("--TYPE", type = str)
#     parser.add_argument("--batch_size", type = int, default=50)
#     parser.add_argument("--epochs", type = int, default = 2)
#     #parser.add_argument("--handicap", type = int, default = 2)
#     parser.add_argument("--img_num", type = int, default = 10)
#
#     args = parser.parse_args()
#     return args
#
# if __name__ == "__main__":
#     #TODO:1)use non-saturating game
#     #TODO:2)use label smoothing
#     #TODO:3)do not apply sigmoid at the output of D
#     #TODO:4)use RMSProp instead of ADAM
#     #TODO:5)lower learning rate,e.g eta=0.00005
#     #TODO:6)include auxiliary information
#     #TODO:7)use more steps for D
#     #TODO:8)use MAD-GAN
#     #TODO:9)move d,g,p inside while
#     # load_image('data128/0.png')
#     train('test64/',batch_size=128,EPOCHS=1200)
#     # generate(500)
#
#     # args = get_args()
#     #
#     # if args.TYPE == 'train':
#     #     train(path = args.path, batch_size = args.batch_size, EPOCHS = args.epochs)
#     #
#     # elif args.TYPE == 'generate':
#     #     generate(img_num = args.img_num)

import argparse
import cv2
import os
import glob
import GAN_models_64 as GAN_models
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
    # print(np.max(img),np.min(img))
    # plt.show()
    # # print(img.shape)
    # # print(np.mean(img,axis=2))
    # print(type(img))
    # img = np.float32((img/ 127.5) - 1)#zero centering the picture
    img = img*2.-1
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

    adam_gen=Adam(lr=0.00001)
    adam_dis=Adam(lr=0.0001)
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
            if os.path.exists('generator_weights_64') and os.path.exists('discriminator_weights_64'):
                print("Loading saves weights..")
                generator.load_weights('generator_weights_64')
                discriminator.load_weights('discriminator_weights_64')
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
        if (epoch+1)%2000==0:
            print('Epoch {} ,Saving weights..'.format(epoch))
            generator.save_weights('generator_weights_64', True)
            discriminator.save_weights('discriminator_weights_64', True)
        if (epoch + 1) % 200 == 0:
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

    generator.load_weights('generator_weights_64')

    noise = np.array( [ generate_code() for _ in range(img_num) ] )

    print('Generating images..')
    generated_images = [img for img in generator.predict(noise)]
    for index, img in enumerate(generated_images):
        cv2.imwrite("{}.jpg".format(index), np.float32(0.5 * (img + 1.0)))

def plot_loss(d,g):
    plt.figure(figsize=(10,8))
    plt.plot(d,label = "discrminitive loss")
    plt.plot(g,label = 'generative loss')
    plt.legend()
    plt.show()

def plot_generated_images(generator,epoch,path ='result_64'):
    # print("saving snapshoot")
    fig = plt.figure()
    Noise_batch = generate_code_batch(9)
    generated_images = generator.predict(Noise_batch)
    print(generated_images.shape)
    plt.clf()
    for i, img in enumerate(generated_images[:9]):
        i = i + 1
        plt.subplot(3, 3, i)
        img = np.float32(0.5 * (img + 1.0))
        plt.imshow(img)
        plt.axis('off')
    fig.canvas.draw()
    path += '/64_Epoch_' + str(epoch) + '.png'
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
    train('test64/',batch_size=128,EPOCHS=6000)
    # generate(2)
    # img = load_image('test64/1.png')
    # plt.imshow(img)
    # plt.show()
    # img = (img+1.)/2
    # plt.imshow(img)
    # plt.show()