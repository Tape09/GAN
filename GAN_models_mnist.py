import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization,Reshape,UpSampling2D
from keras.utils import np_utils
from sklearn.utils import shuffle

def discriminator():
    leakyT = 0.2
    dropT = 0.2
    model = Sequential()
    #1->(32,32,128)
    model.add(Convolution2D(256, (5, 5), strides=(2, 2), input_shape=(28, 28, 1), padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(leakyT))
    model.add(Dropout(dropT))
    #2->(16,16,256)
    model.add(Convolution2D(512, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(leakyT))
    model.add(Dropout(dropT))
    #3->(8,8,512),one filter:5*5*256, 512 filters in total
    # model.add(Convolution2D(512, (3, 3), strides=(1, 1), padding='same'))
    # # model.add(BatchNormalization())
    # model.add(LeakyReLU(0.2))
    # model.add(Dropout(0.2))
    #final
    model.add(Flatten())
    model.add(Dense(units=256))
    model.add(LeakyReLU(leakyT))
    model.add(Dropout(dropT))

    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    return model

def generator(inputdim = 100, xdim=14, ydim=14):
    model = Sequential()
    #pre, 100->1024*4*4
    model.add(Dense(input_dim=inputdim, units=256 * xdim * ydim))
    #1)4*4*1024->8*8*512
    # model.add(BatchNormalization())#batch norm in G can cause strong intra-class correlation
    model.add(Activation('relu'))
    model.add(Reshape((xdim, ydim,256), input_shape=(inputdim,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), padding='same'))
    #2)->16*16*256
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # # model.add(UpSampling2D(size=(2, 2)))
    # model.add(Convolution2D(64, (3, 3), padding='same'))
    #3->32*32*128
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, (1, 1), padding='same'))

    #final
    # model.add(Activation('tanh'))
    model.add(Activation('sigmoid'))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def main():
    D = discriminator()
    G = generator()
    model = generator_containing_discriminator(G,D)
    G.summary()
    D.summary()
    model.summary()

if __name__=="__main__":
    main()
