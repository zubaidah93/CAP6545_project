import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import scipy
from scipy.io import savemat
from sklearn import preprocessing
from scipy.io import loadmat
from tensorflow.keras.callbacks import EarlyStopping
# this sampling layer is the bottleneck layer of variational autoencoder,
# it uses the output from two dense layers z_mean and z_log_var as input, 
# convert them into normal distribution and pass them to the decoder layer

# direc = r'dataset'     #data directory
# files = [f for f in os.listdir(direc) if f.endswith('.mat')]

# for j, name in enumerate(files):
#     filename = os.path.join(direc, name)
#     data = scipy.io.loadmat(filename)
    # Assuming that the data you need is stored in a variable 'data' in the .mat file
train_data = []
train_label = []



folder = 'dataset'
files = os.listdir(folder)
for file in files:
	filename = os.path.join(folder, file)    
	mat = scipy.io.loadmat(filename)
	data = mat['data']
	data_time = data[:32, :, :]
	data_label = data[32, 0, :32]
	for i in range(data_time.shape[0]):
		train_data.append(data_time[i,:,:])
		train_label.append(data_label[i])

# Convert lists to numpy arrays for further processing
train_data = np.array(train_data)
train_label = np.array(train_label)


os.makedirs('train_trials', exist_ok=True)
for i in range(train_data.shape[0]):
	savemat(f'train_trials/train_ssvep_{i}.mat', {'data': train_data[i, :, :], 'label': train_label[i]})

# indices_13 = np.where(train_label == 13)[0]
# indices_17 = np.where(train_label == 17)[0]
# indices_21 = np.where(train_label == 21)[0]

# print("Indices of 13: ", indices_13)
# print("Indices of 17: ", indices_17)
# print("Indices of 21: ", indices_21)

print("train data shape " + str(train_data.shape))
print("train label shape " + str(train_label.shape))
# min_val = np.min(train_data)
# max_val = np.max(train_data)

# # Min-Max scaling
# train_data = (train_data - min_val) / (max_val - min_val)

class Sampling(layers.Layer):
	"""Uses (mean, log_var) to sample z, the vector encoding a digit."""
	def call(self, inputs):
		mean, log_var = inputs
		batch = tf.shape(mean)[0]
		dim = tf.shape(mean)[1]
		epsilon = tf.random.normal(shape=(batch, dim))
		return mean + tf.exp(0.5 * log_var) * epsilon

# 80:20 training-validation split
#X_train, X_test = train_test_split(train_data,  test_size=0.2) #this is for binary classification 
X = train_data
y = train_label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
latent_dim = 2
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

encoder_inputs = keras.Input(shape=(8, 1280, 1))
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
mean = layers.Dense(latent_dim, name="mean")(x)
log_var = layers.Dense(latent_dim, name="log_var")(x)
z = Sampling()([mean, log_var])
encoder = keras.Model(encoder_inputs, [mean, log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(2 * 320 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((2, 320, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x) 
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
	def __init__(self, encoder, decoder, **kwargs):
		super().__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.reconstruction_loss_tracker = keras.metrics.Mean(
			name="reconstruction_loss"
		)
		self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.reconstruction_loss_tracker,
			self.kl_loss_tracker,
		]

	def train_step(self, data):
		with tf.GradientTape() as tape:
			mean,log_var, z = self.encoder(data)
			reconstruction = self.decoder(z)
			reconstruction_loss = tf.reduce_mean(
				tf.reduce_sum(
					keras.losses.binary_crossentropy(data, reconstruction),
					axis=(1, 2),
				)
			)
			kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
			kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
			total_loss = reconstruction_loss + kl_loss 
		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.total_loss_tracker.update_state(total_loss)
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		self.kl_loss_tracker.update_state(kl_loss)
		return {
			"loss": self.total_loss_tracker.result(),
			"reconstruction_loss": self.reconstruction_loss_tracker.result(),
			"kl_loss": self.kl_loss_tracker.result(),
		}


X_train = X_train.astype('float32') / np.max(X_train)
X_test = X_test.astype('float32') / np.max(X_test)

# Add an extra dimension to X_train and X_test
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
class VAE(keras.Model):
	def __init__(self, encoder, decoder, **kwargs):
		super(VAE, self).__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder

	def call(self, inputs):
		z_mean, z_log_var, z = self.encoder(inputs)
		decoded = self.decoder(z)
		return decoded
# Now you can fit your VAE model
vae = VAE(encoder, decoder)
#vae.summary()
#vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999))
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999), 
			loss=keras.losses.MeanSquaredError())
#vae.fit(X_train, epochs=1000, batch_size=128)
# early stopping callback
callbacks = EarlyStopping(monitor = 'val_loss',
                          mode='min',
                          patience =50,
                          verbose = 1,
                          restore_best_weights = True)

# fit vae model
history = vae.fit(X_train,X_train,
            epochs=1000,
            batch_size=32,
            validation_data=(X_test, X_test),callbacks=callbacks)

# loss curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss curves')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# 2D plot of the classes in latent space
z_m, _, _ = encoder.predict(X_test,batch_size=32)
plt.figure(figsize=(12, 10))
plt.scatter(z_m[:, 0], z_m[:, 1], c=X_test[:,0,0,0])
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()

#random_vector = np.random.normal(size=(1, latent_dim))

generated_data = []


# Encode the test data into the latent space
z_mean, _, _ = encoder.predict(X_test,batch_size=128)

# Decode the encoded data back into the original space
generated_data = decoder.predict(z_mean)

print("this is generated data shape", generated_data.shape)
#print(generated_data)
# Create directory for generated images
os.makedirs('generated_data', exist_ok=True)
for i in range(generated_data.shape[0]):
    savemat(f'generated_data/generated_ssvep_{i}.mat', {'data': generated_data[i, :, :, 0]}, {'label': y_test[i]})

# min_val_train = np.min(X_train)
# max_val_train = np.max(X_train)
# print("this is min val training", min_val_train)
# print("this is max val training", max_val_train)
# # Compute the minimum and maximum of the generated data
# min_val_gen = np.min(generated_data)
# max_val_gen = np.max(generated_data)
# print("this is min val generative", min_val_gen)
# print("this is max val generative", max_val_gen)

# # Apply Min-Max scaling to the generated data
# rescaled_generated_data = (generated_data - min_val_gen) * (max_val_train - min_val_train) / (max_val_gen - min_val_gen) + min_val_train

# min_val = np.min(X_test)
# max_val = np.max(X_test)
# print("this is min val", min_val)
# print("this is max val", max_val)
# rescaled_generated_data = generated_data * (max_val - min_val) + min_val
# Compute the mean and standard deviation of the training data
mean_train = np.mean(X_train)
std_train = np.std(X_train)

# Compute the mean and standard deviation of the generated data
mean_gen = np.mean(generated_data)
std_gen = np.std(generated_data)

# Standardize the generated data
standardized_generated_data = (generated_data - mean_gen) / std_gen * std_train + mean_train
os.makedirs('standarized_generated_data', exist_ok=True)

print("this is standarized generated data shape", standardized_generated_data.shape)
print("this is standarized generated data", standardized_generated_data)
generated_data = np.squeeze(generated_data, axis=-1)
for i in range(standardized_generated_data.shape[0]):
	savemat(f'standarized_generated_data/generated_ssvep_{i}.mat', {'data': standardized_generated_data[i, :, :], 'label': y_test[i]})
# Plot the generated images
def plot_label_clusters(encoder, decoder, data, test_lab):
	print("this is data shape inside loop", data.shape)
	if len(data.shape) > 2:
		data = np.squeeze(data, -1)
	print("this is data shape after squeeze", data.shape)
	z_mean, _, _ = encoder.predict(data[..., np.newaxis])
	plt.figure(figsize =(4, 4))
	sc = plt.scatter(z_mean[:, 0], z_mean[:, 1], c = test_lab, cmap='viridis')
	unique_labels = np.unique(test_lab)
	cbar = plt.colorbar(sc, ticks = unique_labels)
	cbar.ax.set_yticklabels(unique_labels)  # Modified line
	plt.xlabel("z[0]")
	plt.ylabel("z[1]")
	plt.show()



labels = {0 :"13",
1: "17",
2: "21"}
# norma_data = []
# for i in range(25):
# 	norma_data.append(X_train[i, :, :])
# Reshape the data to be 2D
# norma_data = np.array(norma_data)
# min_val = np.min(norma_data)
# max_val = np.max(norma_data)
# Min-Max scaling
#norma_data = (norma_data - min_val) / (max_val - min_val)

x_train = np.squeeze(np.expand_dims(X_train, -1).astype("float32"), -1) #/ norma_data
plot_label_clusters(encoder, decoder, x_train, y_train)

print(vae.optimizer.learning_rate)

import numpy as np
import matplotlib.pyplot as plt

def plot_fft(data, title):
    # Compute the FFT
    fft = np.fft.fft(data)
    # Compute the frequencies associated with the FFT values
    freq = np.fft.fftfreq(len(data))
    # Plot the absolute value of the FFT
    plt.figure(figsize=(6, 4))
    plt.plot(freq, np.abs(fft))
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

# Select one trial, one channel from the generated data
generated_trial_channel = standardized_generated_data[2, 1, :]

# Select one trial, one channel from the training data
train_trial_channel = X_train[2, 1, :]

# Plot the FFT of the selected trial and channel from the generated data
plot_fft(generated_trial_channel, 'FFT of Generated Data')

# Plot the FFT of the selected trial and channel from the training data
plot_fft(train_trial_channel, 'FFT of Training Data')

def plot_time_series(data, title):
    # Plot the time series data
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# Select one trial, one channel from the generated data
generated_trial_channel = standardized_generated_data[2, 1, :]

# Select one trial, one channel from the training data
train_trial_channel = X_train[2, 1, :]

# Plot the time series of the selected trial and channel from the generated data
plot_time_series(generated_trial_channel, 'Time Series of Generated Data')

# Plot the time series of the selected trial and channel from the training data
plot_time_series(train_trial_channel, 'Time Series of Training Data')