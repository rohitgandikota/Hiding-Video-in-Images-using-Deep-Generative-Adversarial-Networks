# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:43:14 2019

@author: rohit
"""
from keras.layers import *
from keras import backend as K
import tensorflow as tf
from keras.models import Model
import numpy as np 
from keras.optimizers import Adam, RMSprop
import datetime
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import cv2
import random
from keras.preprocessing import image
from pretrained import VGG19
from keras.models import model_from_json
from sklearn.utils import shuffle
from skimage.measure import compare_ssim as ssim
from PIL import Image

def frame_disc():
    df = 8
    frames = 16
    img_shape = (frames,64,64,3)
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = TimeDistributed(Conv2D(filters//2, kernel_size=f_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01)))(layer_input)
        d = TimeDistributed(LeakyReLU(alpha=0.2))(d)
        d = TimeDistributed(Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))(d)
        d = TimeDistributed(LeakyReLU(alpha=0.2))(d)
        if bn:
            d = TimeDistributed(BatchNormalization(momentum=0.8))(d)
        res = TimeDistributed(Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01)))(layer_input)
        d = add([d, res])
        return d
    
    img_A = Input(shape=img_shape)

    # Concatenate image and conditioning image by channels to produce input

    d1 = d_layer(img_A, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)
    d5 = d_layer(d4, df*8)
    d6 = d_layer(d5, df*8)
    d7 = TimeDistributed(Flatten())(d6)
    d8 = TimeDistributed((Dense(1,activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))))(d7)
    disc = Model(img_A,d8)
    return disc

def vid_disc():
    df = 4
    frames = 16
    img_shape = (frames,64,64,3)
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv3D(filters//2, kernel_size=f_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(d)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        res = Conv3D(filters, kernel_size=f_size, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = add([d, res])
        return d

    img_A = Input(shape=img_shape)

    # Concatenate image and conditioning image by channels to produce input
    d1 = d_layer(img_A, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)
    d5 = d_layer(d4, df*8)
    flat = Flatten()(d5)
    validity = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(flat)
    disc = Model(img_A,validity)
    return disc

def extractor():
    """U-Net Generator"""
    gf = 4
    channels = 3
    frames= 16
    img_shape = (64,64,3)
    
   
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters//2, kernel_size=f_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(d)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        res = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = add([d, res])
        return d
    def conv3d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv3D(filters//2, kernel_size=f_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv3D(filters, kernel_size=f_size, strides=(1,2,2), padding='same', kernel_regularizer=regularizers.l2(0.01))(d)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        res = Conv3D(filters, kernel_size=f_size, strides=(1,2,2), padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = add([d, res])
        return d

    def deconv3d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u1 = UpSampling3D(size=(1,2,2))(layer_input)
        u = Conv3D(filters//2, kernel_size=f_size, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(u1)
        u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        res = Conv3D(filters*2, kernel_size=f_size, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(u)
        u = add([u, res])
        return u

    # Image input
    inp = Input(shape=img_shape)
    proc = conv2d(inp, frames*3)
    d0 = Reshape((frames,64,64,3))(proc)
    # Downsampling
    d1 = conv3d(d0, gf, bn=False)
    d2 = conv3d(d1, gf*2)
    d3 = conv3d(d2, gf*4)
    d4 = conv3d(d3, gf*8)
    d5 = conv3d(d4, gf*16)


    u3 = deconv3d(d5, d4, gf*8)
    u4 = deconv3d(u3, d3, gf*4)
    u5 = deconv3d(u4, d2, gf*2)
    u6 = deconv3d(u5, d1, gf)

    u7 = UpSampling3D(size=(1,2,2),name='up6')(u6)
    output_vid = Conv3D(channels, kernel_size=4, strides=1, padding='same', activation='sigmoid',name='conv6', kernel_regularizer=regularizers.l2(0.01))(u7)
    ex = Model(inp,output_vid)    
    return ex
def build_generator():
    """U-Net Generator"""
    gf = 16
    channels = 3
    timesteps=16
    img_shape = (64,64,3)
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters//2, kernel_size=f_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(d)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        res = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = add([d, res])
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u1 = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters//2, kernel_size=f_size, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(u1)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        res = Conv2D(filters*2, kernel_size=f_size, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(u)
        u = add([u, res])
        return u

    # Image input
    msg = Input(shape=(timesteps, 64,64,3))
    cover = Input(shape=(64,64,3))
    x_cover = Conv2D(8,(3,3),padding='same')(cover)
    
    x = ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', 
                         return_sequences= True)(msg)
    x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', 
                         return_sequences= True)(msg)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', 
                         return_sequences= False)(msg)
    x = Concatenate(axis=-1)([x,x_cover])
    d0 = Concatenate(axis=-1)([x,x_cover])
    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*16)
#    d6 = conv2d(d5, gf*32)
#    d7 = conv2d(d6, gf*16)
    # Upsampling
#    u1 = deconv2d(d7, d6, gf*8)
#    u2 = deconv2d(d6, d5, gf*16)
    u3 = deconv2d(d5, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2,name='up6')(u6)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='sigmoid',name='conv6', kernel_regularizer=regularizers.l2(0.01))(u7)
    gen = Model([cover,msg],output_img)

    return gen


def build_discriminator():
    df = 4
    img_shape = (64,64,3)
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_A = Input(shape=img_shape)

    # Concatenate image and conditioning image by channels to produce input

    d1 = d_layer(img_A, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)
    d5 = d_layer(d4, df*8)
    d6 = d_layer(d5, df*8)
    flat = Flatten()(d6)
    
    
    validity = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(flat)
    disc = Model(img_A,validity)
    return disc
#%%%%%%  Training procedure
optimizer = Adam(0.0002, 0.5)
batch_size = 1
ex = extractor()
D1 = vid_disc()
D2 = frame_disc()
D1.compile(loss='mse',
    optimizer=optimizer,
    metrics=['accuracy'])
D2.compile(loss='mse',
    optimizer=optimizer,
    metrics=['accuracy'])

valid_vid = np.ones((batch_size,))
valid_frame = np.ones((batch_size,16,1))
fake_vid = np.zeros((batch_size,))
fake_frame = np.zeros((batch_size,16,1))

def myFunc(x):
    return x[:,1:]- x[:,:-1]

img_A  = Input(shape=(64,64,3))
vid_A  = ex(img_A)
motion_vid = Lambda(myFunc, output_shape= (15,64,64,3))(vid_A)
D1.trainable = False
D2.trainable = False
valido_vid = D1(vid_A)
valido_frame = D2(vid_A)

extractor = Model(inputs=img_A, outputs=[valido_vid, valido_frame, vid_A, motion_vid])
extractor.compile(loss=['binary_crossentropy', 'mse', 'mse', 'mse'],
                      loss_weights=[1, 1, 100, 100],
                      optimizer= optimizer)


batch_size = 1
cover_shape = (64,64,3)
msg_shape = (16,64,64,3)


# Adversarial loss ground truths
valid = np.ones((batch_size,))
fake = np.zeros((batch_size,))

optimizer = Adam(0.0002, 0.5)


# Build and compile the discriminator

disc = build_discriminator()
disc.compile(loss='mse',
    optimizer=optimizer,
    metrics=['accuracy'])

#-------------------------
# Construct Computational
#   Graph of Generator
#-------------------------

gen = build_generator()
# Input images and their conditioning images
#ex.compile(loss='mse', optimizer=optimizer1)

img_A = Input(shape=cover_shape)
img_B = Input(shape = msg_shape)
# By conditioning on B generate a fake version of A
fake_A = gen([img_A,img_B])

#Generating spectrogram

# For the combined model we will only train the generator
disc.trainable = False
#    disc_feat.trainable = False
# Discriminators determines validity of translated images / condition pairs
valido = disc(fake_A)
#    valido_feat = disc_feat(spect)
model = VGG19(include_top=False, weights='imagenet')
model.trainable = False
feature = model(fake_A)

combined = Model(inputs=[img_A,img_B], outputs=[valido, fake_A,feature])
combined.compile(loss=['mse', 'mae','mse'],
                      loss_weights=[1, 100,100],
                      optimizer=optimizer)



images_path = "./DIV2K/train/DIV2K_train_HR/DIV2K_train_HR"
images_name = os.listdir(images_path)
datasetlist = []
numSamples = 1
cnt = 0
for filename in images_name:
    cnt += 1
    if cnt==numSamples+1:
        break
    imag = Image.open(images_path+'/'+filename)
    imag = imag.resize([64,64], Image.ANTIALIAS)
    imag = (np.array(imag) - 127.5 / 127.5)
    datasetlist.append(imag)
    f = open('3dgan/imgs.txt','a+')
    f.write( ' \n'+ filename)
    f.close()

x = np.array(datasetlist)
  
for i in range(x.shape[0]):
    x[i] = (x[i]-x[i].min())/(x[i].max()-x[i].min())
X = x 

loc = "./DATA/"
vid_paths = os.listdir(loc)
x1 = np.zeros((50,64,64,3))
x1 = np.stack((x1,x1))
for path in vid_paths:
    frames = os.listdir(loc+path+'/img/')
    vid = []
    cnt = 0
    print('Inputting video')
    for i in range(50):
        if i < 9:
            imag = Image.open(loc+path+'/img/000'+str(i+1)+'.jpg')
        if i>8:
            imag = Image.open(loc+path+'/img/00'+str(i+1)+'.jpg')        
        imag = imag.resize([64,64], Image.ANTIALIAS)
        imag = (np.array(imag) - 127.5 / 127.5)
        vid.append(imag)
        cnt+=1
        if cnt == 50:
            break
    vid = np.array(vid)
    vid = np.expand_dims(vid, axis=0)
    vid = (vid - vid.min()) / (vid.max() - vid.min())
    if np.shape(vid) == (1,50,64,64,3):
        x1 = np.vstack([x1,vid])
x1 = x1[2:]
x1 = x1[:1]
from scipy.interpolate import interp1d
def resample(x, factor, kind='linear'):
    n = np.ceil(x.size / factor)
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))
a = np.arange(50)
vid = resample(a, factor= 50/16 , kind='nearest')
vid = np.int32(vid)
x1 = x1[:,vid]
x1_motion = x1[:,1:] - x1[:,:-1]
epochs = 100000
feature_x = model.predict(X)

select1 = []
select2 = []
Gloss = []
Eloss = []
Dloss = []

start_time = datetime.datetime.now()
for epoch in range(epochs):
    # ---------------------
    #  Train Discriminator
    # ---------------------
    # Condition on B and generate a translated version
    fake_img = gen.predict([X,x1])
    fake_vid = ex.predict(fake_img)
    fake_motion = extractor.predict(fake_img)[-1]
    # Train the discriminators (original images = real / generated = Fake)
    d_loss_real = disc.train_on_batch(X, valid)
    d_loss_fake = disc.train_on_batch(fake_img, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    d1_loss_real = D1.train_on_batch(x1, valid_vid)
    d1_loss_fake = D1.train_on_batch(fake_vid, fake)
    d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
    
    
    d2_loss_real = D2.train_on_batch(x1, valid_frame)
    d2_loss_fake = D2.train_on_batch(fake_vid, fake_frame)
    d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)
    # -----------------
    #  Train Generator
    # -----------------

    # Train the generators
 
    g_loss = combined.fit(x=[X,x1],y=[valid,X,feature_x],batch_size=15,epochs=1,verbose=False)
#    g_loss = combined.train_on_batch( [X,x1] , [valid, X,feature_x])
#        f_loss = featured.train_on_batch(x,feature_x)
    extractor_loss = extractor.fit(x=fake_img, y = [valid_vid, valid_frame, x1, x1_motion], batch_size=30,epochs= 1,verbose= False) 

    elapsed_time = datetime.datetime.now() - start_time
    Gloss.append(g_loss.history['loss'][-1])
    Dloss.append([d_loss[0],d1_loss[0], d2_loss[0]])
    Eloss.append(extractor_loss.history['loss'][-1])
    # Plot the progress
    print ("[Epoch %d/%d]  [D loss: %f, acc: %3d%%] [G loss: %f] [E loss: %f] time: %s" % (epoch, epochs,
                                                            d_loss[0], 100*d_loss[1],
                                                            g_loss.history['loss'][-1], extractor_loss.history['loss'][-1],
                                                                elapsed_time))
    if (epoch%100 ==0) :
        
        if (epoch%1000 ==0) :
            gen.save_weights("./3dgan/Models/gen"+str(epoch)+".h5")
            print("Saved generator model to disk")
            ex.save_weights("./3dgan/Models/ex"+str(epoch)+".h5")
            print("Saved extractor model to disk")
            select1.append(extractor_loss.history['loss'][-1])
            select2.append(g_loss.history['loss'][-1])
            # from 8500 epochs it get's started
            
        num = 0
        img = X[num]
        img =  np.expand_dims(img, axis=0)
        vid = x1[num]
        vid =  np.expand_dims(vid, axis=0)
        fig = plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        g_img = gen.predict([img,vid])
#            plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(g_img[0])
        plt.axis('off')
        plt.title('Generated image')

        plt.subplot(2,2,2)
#            plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(img[0])
        plt.axis('off')
        plt.title('Actual Cover Image')
        plt.subplot(2,2,3)
        e_img = ex.predict(gen.predict([img,vid]))
#        e_img = (e_img-e_img.min()) / (e_img.max()-e_img.min())
#            plt.imshow(cv2.cvtColor(g_img[0], cv2.COLOR_RGB2BGR))
#        e_img = e_img -0.5
        num1 = random.randint(0,15)
        plt.imshow(e_img[0][num1])
        
        plt.axis('off')
        plt.title('extracted message image')

        plt.subplot(2,2,4)
#            plt.imshow(cv2.cvtColor(c_img[0], cv2.COLOR_RGB2BGR))
        plt.imshow(vid[0][num1])
        plt.axis('off')
        plt.title('Actual message Image')
        
    
        fig.savefig('./3dgan/Test_epoch'+str(epoch)+'.png')

Dloss = np.array(Dloss)
fig = plt.figure(figsize=(25,5))

plt.subplot(1,5,1)
plt.plot(Gloss)
plt.grid(True)    
plt.title('Generator Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.subplot(1,5,2)
plt.plot(Dloss[:,0])
plt.grid(True)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Generator Discriminator Loss')

plt.subplot(1,5,3)
plt.plot(Dloss[:,1])
plt.grid(True)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Video Discriminator Loss')

plt.subplot(1,5,4)
plt.plot(Dloss[:,2])
plt.grid(True)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Frame Discriminator Loss')

plt.subplot(1,5,5)
plt.plot(Eloss)
plt.grid(True)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title("Extractor Loss")  
fig.savefig('./3dgan/loss_curves.png')


arg = np.argmin(select1)
g_model_arg = arg*1000
gen.load_weights('./3dgan/Models/gen'+str(g_model_arg)+'.h5')
arg = np.argmin(select2)
model_arg = arg*1000
ex.load_weights('./3dgan/Models/ex'+str(model_arg)+'.h5')



enc_imgs = gen.predict([X,x1],batch_size=15)
ex_vids = ex.predict(enc_imgs,batch_size=15)

#enc_imgs = (enc_imgs - enc_imgs.min() )/ (enc_imgs.max() - enc_imgs.min())
#ex_imgs = (ex_imgs - ex_imgs.min() )/ (ex_imgs.max() - ex_imgs.min())

mse_ci = np.mean((X-enc_imgs)**2)
mse_ex = np.mean((x1-ex_vids)**2)

psnr_ci = 10*np.log10(1/mse_ci)
psnr_ex = 10*np.log10(1/mse_ex)

woow = []
for i in range(len(enc_imgs)):
    ssim_ci = ssim(X[i].reshape(64,64,3),enc_imgs[i].reshape(64,64,3).astype('float64'), data_range = enc_imgs[i].max() - enc_imgs[i].min(),multichannel=True)
    woow.append(ssim_ci)
    
# file-output.py
f = open('3dgan/results_video.txt','a+')
f.write('Best embedder Model:'+ str(g_model_arg) + ' \n'+'Best extractor Model:'+ str(model_arg) + ' \n'+ ' PSNR_CI = '+ str(psnr_ci)+ '\n' + ' PSNR_ex = ' +str(psnr_ex) + '\n' + ' SSIM = ' +str(np.mean(woow))+'\n')
f.close()


enc_imgs = gen.predict([X,x1],batch_size=15)
vid_pred = ex.predict(enc_imgs,batch_size=15) 
motion_vid = x1_motion
vid_pred = vid_pred *255
motion_vid = motion_vid*255
vid_act = x1 *255

def printVid(path,num_vid,vid_pred,vid_act,motion_vid):
    for i in range(num_vid):
        out_pred = cv2.VideoWriter(path+'Prediction'+str(i+1)+'.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,64), 3)
        out_mot = cv2.VideoWriter(path+'Motion'+str(i+1)+'.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,64), 3)
        out_act  = cv2.VideoWriter(path+'Actual'+str(i+1)+'.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,64), 3)
        for j in range(len(vid_pred[i])):
#             Write the frame into the file 'output.avi'
            out_pred.write(np.uint8(vid_pred[i][j].reshape(64,64,3)))
            out_act.write(np.uint8(vid_act[i][j].reshape(64,64,3)))
            if j!=15:
                out_mot.write(np.uint8(motion_vid[i][j].reshape(64,64,3)))
        # When everything done, release the video capture and video write object 
        out_pred.release()
        out_act.release()
        out_mot.release()
    return None 

printVid('./3dgan/Results/',1,vid_pred,vid_act,motion_vid)
