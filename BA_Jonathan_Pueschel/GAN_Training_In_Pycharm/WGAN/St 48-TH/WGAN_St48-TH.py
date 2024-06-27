import pandas as pd
from keras.src.layers import LeakyReLU, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import backend


# Read labeled_OES_data_2.csv
data = pd.read_csv("../labeled_OES_data_2.csv")

# convert steel types to numbers
# Note: in our project steel types are not predefined, but will be extracted from the supplied file of data
classes_dict = {"St 48" : 0, "St 37" : 1, "St Si" : 2, "St 48-TH" : 3, "Puddelstahl" : 4}
data["steel_class"] = [classes_dict[c] for c in data["steel_class"] ]

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
    wgan = Sequential()
    wgan.add(generator)
    wgan.add(critic)
    return wgan

def weight_clipping(weights, weight_clipping_value):
    return backend.clip(weights, -weight_clipping_value, weight_clipping_value)


def wasserstein_loss_gen(fake_output):
    return -tf.reduce_mean(fake_output)

def wasserstein_loss_critic(real_output, fake_output):
    return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))


latent_space = 20
optimizer_critic = RMSprop(learning_rate=0.0005)
optimizer_wgan = RMSprop(learning_rate=0.0005)

critic_St48_TH = build_critic()
critic_St48_TH.compile(optimizer=optimizer_critic, loss=wasserstein_loss_critic)

generator_St48_TH = build_generator(latent_space)
generator_St48_TH.summary()

wgan_St48_TH = build_wgan(generator_St48_TH, critic_St48_TH)
wgan_St48_TH.compile(optimizer=optimizer_wgan, loss=wasserstein_loss_gen)

epochs = 2500
batch_size = 64
#learning_rate = 0.00005
weight_clipping_value = 0.01
iteration_critic = 5

X_train_St48_TH = dfs_by_steeltype[3]

c_real_hist = []
wgan_hist = []

for epoch in range(epochs):
    c_real_iteration = []
    c_fake_iteration = []
    for iteration in range(iteration_critic):

        real_samples = X_train_St48_TH.values[np.random.randint(0, X_train_St48_TH.shape[0], batch_size)]

        lat_noise = np.random.normal(0, 1, (batch_size, latent_space))
        fakes = generator_St48_TH.predict(lat_noise)

        # Train the critic
        with tf.GradientTape() as tape:
            real_output = critic_St48_TH(real_samples, training=True)
            fake_output = critic_St48_TH(fakes, training=True)
            c_loss = wasserstein_loss_critic(real_output, fake_output)
        gradients = tape.gradient(c_loss, critic_St48_TH.trainable_variables)
        #print("Critic Gradient: ", gradients)
        #print("Critic weights: ", critic_St48_TH.get_weights())
        optimizer_critic.apply_gradients(zip(gradients, critic_St48_TH.trainable_variables))

        for layer in critic_St48_TH.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -weight_clipping_value, weight_clipping_value) for w in weights]
            layer.set_weights(weights)

    lat_noise = np.random.normal(0, 1, (batch_size, latent_space))

    with tf.GradientTape() as tape:
        generated_data = generator_St48_TH(lat_noise, training=True)
        fake_output = critic_St48_TH(generated_data, training=False)
        g_loss = wasserstein_loss_gen(fake_output)
    gradients = tape.gradient(g_loss, generator_St48_TH.trainable_variables)
    #print("Generator Gradient: ", gradients)
    #print("Generator weights: ", generator_St48_TH.get_weights())
    optimizer_wgan.apply_gradients(zip(gradients, generator_St48_TH.trainable_variables))
    #print()
    print(f'Epoch: {epoch + 1}, Critic Loss: {c_loss.numpy()}, Generator Loss: {g_loss.numpy()}')
    wgan_hist.append(g_loss)
    c_real_hist.append(c_loss)


plt.plot(c_real_hist, label='critic loss')
plt.plot(wgan_hist, label='gen loss')
plt.legend()
#plt.savefig('WGAN_loss_fig_St48_TH_2500.png')
plt.show()


#Save the generator model
#generator_St48_TH.save("WGAN_generator_St48_TH_2500Epochs.keras")

# Save the discriminator model
#critic_St48_TH.save("WGAN_critic_St48_TH_2500Epochs.keras")





