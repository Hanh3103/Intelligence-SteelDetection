import keras.src.saving
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.src.layers import LeakyReLU, BatchNormalization, Activation
from keras.src.optimizers import RMSprop
from keras import backend as K

class WGAN_steel_type_data_augmentation:
    # HYPERPARAMETER
    latent_space = 20
    epochs = 5
    batch_size = 64
    weight_clipping_value = 0.1
    iteration_critic = 5

    def __init__(self):
        self.optimizer_critic = RMSprop(learning_rate=0.0005)
        self.optimizer_wgan = RMSprop(learning_rate=0.0005)
        self.generator = self.build_generator(latent_space= self.latent_space)
        self.critic = self.build_critic()
        self.critic.compile(optimizer=self.optimizer_critic, loss=self.wasserstein_loss_critic)
        self.wgan = Sequential()
        self.wgan.add(self.generator)
        self.wgan.add(self.critic)
        self.wgan.compile(optimizer=self.optimizer_wgan, loss=self.wasserstein_loss_gen)

    def build_generator(self, latent_space):
        generator = Sequential()
        generator.add(Dense(latent_space, input_dim=latent_space))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Dense(256))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Dense(512))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(BatchNormalization(momentum=0.8))
        generator.add(Dense(9))
        generator.add(Activation('sigmoid'))
        return generator


    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(10, input_dim=9))
        critic.add(Dense(512))
        critic.add(LeakyReLU(alpha=0.2))
        critic.add(Dropout(0.3))
        critic.add(Dense(256))
        critic.add(LeakyReLU(alpha=0.2))
        critic.add(Dropout(0.3))
        critic.add(Dense(128))
        critic.add(LeakyReLU(alpha=0.2))
        critic.add(Dense(1))
        return critic


    def wasserstein_loss_gen(self, fake_output):
        return -tf.reduce_mean(fake_output)


    def wasserstein_loss_critic(self, real_output, fake_output):
        return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))

    # replaced bc of error
    def weight_clipping(weights, weight_clipping_value):
        return K.clip(weights, -weight_clipping_value, weight_clipping_value)

    def train_wgan(self, data, steel_type):
        #self.optimizer_critic = RMSprop(learning_rate=0.0005)
        for epoch in range(self.epochs):
            for iteration in range(self.iteration_critic):

                real_samples = data.values[np.random.randint(0, data.shape[0], self.batch_size)]

                lat_noise = np.random.normal(0, 1, (self.batch_size, self.latent_space))
                fakes = self.generator.predict(lat_noise)

                # Train the critic
                with tf.GradientTape() as tape:
                    real_output = self.critic(real_samples, training=True)
                    fake_output = self.critic(fakes, training=True)
                    c_loss = self.wasserstein_loss_critic(real_output, fake_output)
                gradients = tape.gradient(c_loss, self.critic.trainable_variables)
                self.optimizer_critic.apply_gradients(zip(gradients, self.critic.trainable_variables))

                for layer in self.critic.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(w, -self.weight_clipping_value, self.weight_clipping_value) for w in weights]
                    layer.set_weights(weights)

            # train generator
            lat_noise = np.random.normal(0, 1, (self.batch_size, self.latent_space))
            with tf.GradientTape() as tape:
                generated_data = self.generator(lat_noise, training=True)
                fake_output = self.critic(generated_data, training=False)
                g_loss = self.wasserstein_loss_gen(fake_output)
            gradients = tape.gradient(g_loss, self.generator.trainable_variables)
            self.optimizer_wgan.apply_gradients(zip(gradients, self.generator.trainable_variables))
        self.generator.save(f"WGAN_generator{steel_type}.keras")
        #self.critic.save(f"WGAN_critic{steel_type}.keras")


    def augment_data(self, amount, data, steel_type):
        try:
            self.generator = keras.models.load_model(f"WGAN_generator{steel_type}.keras")
        except:
                self.train_wgan(data, steel_type)

        lat_noise = np.random.normal(0, 1, (amount, self.latent_space))
        return self.generator.predict(lat_noise)