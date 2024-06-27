import random

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.src.layers import LeakyReLU, BatchNormalization, Activation
from keras.src.optimizers import Adam

# Read labeled_OES_data_2.csv
data = pd.read_csv("../labeled_OES_data_2.csv")

# convert steel types to numbers
# Note: in our project steel types are not predefined, but will be extracted from the supplied file of data
classes_dict = {"St 48": 0, "St 37": 1, "St Si": 2, "St 48-TH": 3, "Puddelstahl": 4}
data["steel_class"] = [classes_dict[c] for c in data["steel_class"]]

# make data numeric
for column in data.columns:
    for index, value in data[column].items():
        if isinstance(value, str):
            data.at[index, column] = value.replace(",", ".")
data = data.map(pd.to_numeric)

element_data_grouped_by_steeltype = data.groupby("steel_class")
dfs_by_steeltype = {}

for type, group_data in element_data_grouped_by_steeltype:
    dfs_by_steeltype[type] = group_data.iloc[:, :-1]


# latent space is random noise out of which the data will be formed
def build_generator(latent_space):
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


def build_critic():
    critic = Sequential()
    critic.add(Dense(9, input_dim=9))
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


def build_wgan(generator, critic):
    # critic.trainable = False
    wgan = Sequential()
    wgan.add(generator)
    wgan.add(critic)
    return wgan


def wasserstein_loss_gen(fake_output):
    return -tf.reduce_mean(fake_output)


def wasserstein_loss_critic(real_output, fake_output):
    return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))


def gradient_penalty_replace(gp_lambda, critic, real, fake):
    # interpolation
    interpolation = real.copy()
    rand_epsilon_length = random.randint(0, 8)
    rand_index = random.randint(0, 8)
    epsilon = []
    while len(epsilon) < rand_epsilon_length:
        if rand_index not in epsilon:
            epsilon.append(rand_index)
        rand_index = random.randint(0, 8)

    for index in epsilon:
        interpolation[0][index] = fake[0][index]

    interpolation_tensor = tf.constant(interpolation, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(interpolation_tensor)
        c_output = critic(interpolation_tensor)
    gradient = tape.gradient(c_output, interpolation_tensor)
    norm_grad = tf.norm(gradient, axis=1)
    return gp_lambda * tf.reduce_mean((norm_grad - 1) ** 2)


def gradient_penalty_mix(gp_lambda, critic, real, fake):
    # interpolation
    interpolation = real.copy()
    epsilon = random.random()
    for index in range(9):
        interpolation[0][index] = epsilon * interpolation[0][index] + (1 - epsilon) * fake[0][index]

    interpolation_tensor = tf.constant(interpolation, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(interpolation_tensor)
        c_output = critic(interpolation_tensor)
    gradient = tape.gradient(c_output, interpolation_tensor)
    norm_grad = tf.norm(gradient, axis=1)
    return gp_lambda * tf.reduce_mean((norm_grad - 1) ** 2)


latent_space = 20
optimizer_critic = Adam(
    learning_rate=1e-4,
    beta_1=0,
    beta_2=0.9
)
optimizer_wgan = Adam(
    learning_rate=1e-4,
    beta_1=0,
    beta_2=0.9
)

critic_St37 = build_critic()
critic_St37.compile(optimizer=optimizer_critic, loss=wasserstein_loss_critic)

generator_St37 = build_generator(latent_space)

wgan_St37 = build_wgan(generator_St37, critic_St37)
wgan_St37.compile(optimizer=optimizer_wgan, loss=wasserstein_loss_gen)

epochs = 1000
batch_size = 64
# learning_rate = 0.00005
iteration_critic = 5

X_train_St37 = dfs_by_steeltype[1]

c_real_hist = []
wgan_hist = []

for epoch in range(epochs):
    c_real_iteration = []
    c_fake_iteration = []
    for iteration in range(iteration_critic):
        real_samples = X_train_St37.values[np.random.randint(0, X_train_St37.shape[0], batch_size)]

        lat_noise = np.random.normal(0, 1, (batch_size, latent_space))
        fakes = generator_St37.predict(lat_noise)
        for i in range(batch_size):
            # Train the critic
            real_sample = np.expand_dims(real_samples[i], axis=0)
            fake_sample = np.expand_dims(fakes[i], axis=0)
            with tf.GradientTape() as tape:
                real_output = critic_St37(real_sample, training=True)
                fake_output = critic_St37(fake_sample, training=True)
                c_loss = wasserstein_loss_critic(real_output, fake_output) + gradient_penalty_mix(10, critic_St37, real_sample, fake_sample)
        gradients = tape.gradient(c_loss, critic_St37.trainable_variables)
        optimizer_critic.apply_gradients(zip(gradients, critic_St37.trainable_variables))

    lat_noise = np.random.normal(0, 1, (batch_size, latent_space))
    with tf.GradientTape() as tape:
        generated_data = generator_St37(lat_noise, training=True)
        fake_output = critic_St37(generated_data, training=False)
        g_loss = wasserstein_loss_gen(fake_output)
    gradients = tape.gradient(g_loss, generator_St37.trainable_variables)
    optimizer_wgan.apply_gradients(zip(gradients, generator_St37.trainable_variables))

    print(f'Epoch: {epoch + 1}, Critic Loss: {c_loss.numpy()}, Generator Loss: {g_loss.numpy()}')
    wgan_hist.append(g_loss)
    c_real_hist.append(c_loss)

plt.plot(c_real_hist, label='critic loss')
plt.plot(wgan_hist, label='gen loss')
plt.legend()
#plt.savefig('WGAN-GP_loss_fig.png')
plt.show()
# Save the generator model
#generator_St37.save("WGAN-GP_generator_St37_1000Epochs.keras")

# Save the discriminator model
#critic_St37.save("WGAN-GP_critic_St37_1000Epochs.keras")
