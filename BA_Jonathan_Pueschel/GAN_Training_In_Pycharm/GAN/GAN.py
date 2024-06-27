import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
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

for steel_type, group_data in element_data_grouped_by_steeltype:
    dfs_by_steeltype[steel_type] = group_data.iloc[:, :-1]


# latent space is random noise out of which the data will be formed
def build_generator(latent_space):
    generator = Sequential()
    generator.add(Dense(latent_space, input_dim=latent_space, activation="relu"))
    generator.add(Dense(128, activation="relu"))
    generator.add(Dropout(0.3))
    generator.add(Dense(128, activation="relu"))
    generator.add(Dense(9, activation="sigmoid"))
    return generator


def build_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(10, input_dim=9, activation="relu"))
    discriminator.add(Dense(128, activation="relu"))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(128, activation="relu"))
    discriminator.add(Dense(1, activation="sigmoid"))
    return discriminator


def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    return gan


latent_space = 10
optimizer_dis = Adam(learning_rate=3e-4)
optimizer_gan = Adam(learning_rate=3e-4)

discriminator_St37 = build_discriminator()
discriminator_St37.compile(optimizer=optimizer_dis, loss="binary_crossentropy")

generator_St37 = build_generator(latent_space)
generator_St37.summary()

gan_St37 = build_gan(generator_St37, discriminator_St37)
gan_St37.compile(optimizer=optimizer_gan, loss="binary_crossentropy")

epochs = 1000
batch_size = 64

X_train_St37 = dfs_by_steeltype[1]

misclassified_St37_arrays = [
    [0.108, 0.0097, 0.565, 0.0254, 0.0262, 0.0214, 0.0431, 0.002, 0.121],
    [0.0918, 0.005, 0.296, 0.0245, 0.0176, 0.0104, 0.0337, 0.002, 0.242],
    [0.104, 0.005, 0.318, 0.0271, 0.0252, 0.0118, 0.0322, 0.0045, 0.261],
    [0.108000, 0.009700, 0.565000, 0.025400, 0.026200, 0.021400, 0.043100, 0.002000, 0.121000],
    [0.148667, 0.005533, 0.552167, 0.031317, 0.031917, 0.030283, 0.039033, 0.002000, 0.119000],
    [0.091800, 0.005000, 0.296000, 0.024500, 0.017600, 0.010400, 0.033700, 0.002000, 0.242000],
    [0.104000, 0.005000, 0.318000, 0.027100, 0.025200, 0.011800, 0.032200, 0.004500, 0.261000]]

misclassified_St37_df = pd.DataFrame(misclassified_St37_arrays)
misclassified_St37_df.columns = ["C", "Si", "Mn", "P", "S", "Cr", "Ni", "Al", "Cu"]

for i in range(7):
    X_train_St37 = pd.concat([X_train_St37, misclassified_St37_df], ignore_index=True)

dis_real_hist = []
dis_fake_hist = []
gen_hist = []
for epoch in range(epochs):
    # get batch_size number of random samples out of training data
    real_samples = X_train_St37.values[np.random.randint(0, X_train_St37.shape[0], batch_size)]

    # get batch_size number of fake samples
    # to speed up computation we chose latent_space = 10
    lat_noise = np.random.normal(0, 1, (batch_size, latent_space))
    fakes = generator_St37.predict(lat_noise)

    discriminator_St37.trainable = True
    dis37_loss_real = discriminator_St37.train_on_batch(real_samples, np.ones((batch_size, 1)))
    dis37_loss_fake = discriminator_St37.train_on_batch(fakes, np.zeros((batch_size, 1)))

    lat_noise = np.random.normal(0, 1, (batch_size, latent_space))
    g37_loss = gan_St37.train_on_batch(lat_noise, np.ones((batch_size, 1)))
    print("Epoch", epoch + 1, ": ", "dis37_loss_fake:", dis37_loss_fake, ",", "dis37_loss_real:", dis37_loss_real, ", ",
          "gen_loss: ", g37_loss)
    dis_real_hist.append(dis37_loss_real)
    dis_fake_hist.append(dis37_loss_fake)
    gen_hist.append(g37_loss)

plt.plot(dis_real_hist, label='discriminator real loss')
plt.plot(dis_fake_hist, label='discriminator fake loss')
plt.plot(gen_hist, label='gen loss')
plt.legend()
# plt.savefig('GAN_loss_fig.png')
plt.show()

# Save the generator model
# generator_St37.save("GAN_generator_St37_1000Epochs.keras")

# Save the discriminator model
# discriminator_St37.save("GAN_discriminator_St37_1000Epochs.keras")
