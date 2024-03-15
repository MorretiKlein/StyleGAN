import tensorflow as tf
import numpy as np
from util.utils import *
from model.generator import Generator
from model.discriminator import Discriminator
from model.Layers import *
from model.StyleGan import StyleGAN

START_RES = 4
TARGET_RES = 128

style_gan = StyleGAN(start_res=START_RES, target_res=TARGET_RES)

def train(start_res=START_RES, target_res=TARGET_RES, steps_per_epoch=5000, display_images=True):
    opt_cfg = {'learning_rate':1e-3, 'beta_1':0.0, 'beta_2':0.99, 'epsilon':1e-8}

    val_batch_size = 16
    val_z = tf.random.normal((val_batch_size, style_gan.z_dim))  
    val_noise = style_gan.generate_noise(val_batch_size)
    
    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2+1):
        res = 2**res_log2
        for phase in ['TRANSITION', 'STABLE']:
            if res==start_res and phase=='TRANSITION':
                continue

            train_dl = create_dataloader(res)

            steps = int(train_step_ratio[res_log2] * steps_per_epoch)

            style_gan.compile(d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                              g_optimizer=tf.keras.optimizers.Adam(**opt_cfg), 
                              loss_weights = {'gradient_penalty':10, 'drift':0.001},
                              steps_per_epoch=steps,
                              res=res,
                              phase=phase, run_eagerly=False)

            prefix = f'res_{res}x{res}_{style_gan.phase}'

            ckpt_cb = keras.callbacks.ModelCheckpoint(f'checkpoints/stylegan_{res}x{res}.ckpt', 
                                      save_weights_only=True, verbose=0)
            print(phase)
            style_gan.fit(train_dl, epochs=1, 
                          steps_per_epoch=steps, callback5s=[ckpt_cb])

            if display_images:
                images = style_gan({'z':val_z, 'noise':val_noise, 'alpha':1.0})
                plot_images(images, res_log2)        


def resize_image(resolution, image):
    image = tf.image.resize(
        image, (resolution,resolution), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)/ 127.5 - 1.0

    return image

def create_dataloader(resolution):
    batch_size = batch_sizes[log2(resolution)]
    data_loader = data_train.map(partial(resize_image ,resolution), num_parallel_calls= tf.data.AUTOTUNE)
    data_loader = data_loader.shuffle(3).batch(batch_size, drop_remainder = True).prefetch(1).repeat()
    return data_loader

batch_sizes = {2:16, 3:16, 4:16, 5:16, 6:16, 7:8, 8:4} # 2**8
train_step_ratio = {k:batch_sizes[2] /v for k,v in batch_sizes.items()}
# {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 2.0, 8: 4.0}
data_train = keras.utils.image_dataset_from_directory("trainB1/", label_mode = None, image_size=(256,256), batch_size= None)

train(steps_per_epoch=1000)