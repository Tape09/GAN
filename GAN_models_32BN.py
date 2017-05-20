# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten,LeakyReLU
# from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization,Reshape,UpSampling2D
# from keras.utils import np_utils
# from sklearn.utils import shuffle
#
# def discriminator():
#     model = Sequential()
#     #1
#     # model.add(BatchNormalization())
#     model.add(Convolution2D(256, (5, 5), strides=(2, 2), input_shape=(32, 32, 3), padding='same'))
#     # model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     #2
#     model.add(Convolution2D(512, (5, 5), strides=(2, 2), padding='same'))
#     # model.add(BatchNormalization())
#     model.add(LeakyReLU(0.2))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     #3
#     # model.add(Convolution2D(128, (3, 3), strides=(1, 1), padding='same'))
#     # # model.add(BatchNormalization())
#     # model.add(LeakyReLU(0.2))
#     # # model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Dropout(0.2))
#
#     # 4
#     # model.add(Convolution2D(256, (3, 3), strides=(2, 2), input_shape=(32, 32, 3), padding='same'))
#     # # model.add(BatchNormalization())
#     # model.add(LeakyReLU(0.2))
#     # # model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Dropout(0.2))
#
#     # 5
#     # model.add(Convolution2D(512, (3, 3), strides=(2, 2), padding='same'))
#     # model.add(BatchNormalization(momentum=0.5,epsilon=1e-8))
#     # model.add(LeakyReLU(0.3))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Dropout(0.2))
#
#     # 6
#     # model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='same'))
#     # # model.add(BatchNormalization())
#     # model.add(LeakyReLU(0.2))
#     # # model.add(MaxPooling2D(pool_size=(2, 2)))
#     # model.add(Dropout(0.2))
#     #final
#     model.add(Flatten())
#     model.add(Dense(units=256))
#     model.add(LeakyReLU(0.2))
#     model.add(Dropout(0.25))
#     model.add(Dense(units=1))
#     model.add(Activation('sigmoid'))
#     return model
#
# def generator(inputdim=120, xdim=4, ydim=4):
#     model = Sequential()
#     #pre, 100->1024*4*4
#     model.add(Dense(input_dim=inputdim, units=512 * xdim * ydim,kernel_initializer="random_normal",bias_initializer="random_normal"))
#
#     #1)4*4*1024->8*8*512
#     model.add(BatchNormalization())#batch norm in G can cause strong intra-class correlation
#     model.add(Activation('relu'))
#     model.add(Reshape((xdim, ydim,512), input_shape=(inputdim,)))
#     model.add(UpSampling2D(size=(2, 2)))
#     model.add(Convolution2D(256, (3, 3), padding='same'))
#
#     #2)->16*16*256
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(UpSampling2D(size=(2, 2)))
#     model.add(Convolution2D(128, (3, 3), padding='same'))
#
#     # 2)->16*16*256
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(UpSampling2D(size=(2, 2)))
#     model.add(Convolution2D(64, (3, 3), padding='same'))
#
#     #3->32*32*128
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     # model.add(UpSampling2D(size=(2, 2)))
#     model.add(Convolution2D(3, (1, 1), padding='same'))
#
#     #final
#     model.add(Activation('tanh'))
#     return model
#
#
#
#
#
#
# def generator_containing_discriminator(generator, discriminator):
#     model = Sequential()
#     model.add(generator)
#     discriminator.trainable = False
#     model.add(discriminator)
#     return model
#
# def main():
#     D = discriminator()
#     G = generator()
#     model = generator_containing_discriminator(G,D)
#     G.summary()
#     D.summary()
#     model.summary()
#
# if __name__=="__main__":
#     main()


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization,Reshape,UpSampling2D
from keras.utils import np_utils
from sklearn.utils import shuffle

def discriminator():
    model = Sequential()
    #1->(32,32,128)
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), input_shape=(32, 32, 3), padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    #2->(16,16,256)
    model.add(Convolution2D(128, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    #3->(8,8,512),one filter:5*5*256, 512 filters in total
    model.add(Convolution2D(256, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    #4->(4,4,1024)
    model.add(Convolution2D(512, (3, 3), strides=(2, 2), padding='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    #final
    model.add(Flatten())
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    return model

def generator(inputdim=120, xdim=4, ydim=4):
    model = Sequential()
    #pre, 100->1024*4*4
    model.add(Dense(input_dim=inputdim, units=512 * xdim * ydim))
    #1)4*4*1024->8*8*512
    model.add(BatchNormalization())#batch norm in G can cause strong intra-class correlation
    model.add(Activation('relu'))
    model.add(Reshape((xdim, ydim,512), input_shape=(inputdim,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, (3, 3), padding='same'))
    #2)->16*16*256
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), padding='same'))
    #3->32*32*128
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, (3, 3), padding='same'))

    #final
    model.add(Activation('tanh'))
    return model

    # # 8. Compile model
    # model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    # # 9. Fit model on training data
    # model.fit(X_train, Y_train,batch_size=32, nb_epoch=10, verbose=1)
    #
    # # 10. Evaluate model on test data
    # score = model.evaluate(X_test, Y_test, verbose=0)

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