import numpy as np
import tensorflow.keras as keras

from keras.models import Sequential
from data_generator import DataGenerator

# Parameters
params = {'dim': (512,512),
          'batch_size': 64,
          'n_classes': 3,
          'n_channels': 487,
          'shuffle': True}

# Orginal Datasets
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

# Change label values to categorical 
vals = np.fromiter(labels.values(), dtype=float)
vals_cat = keras.utils.to_categorical(vals, num_classes=3, dtype='float32')
for i, k in enumerate(labels):
    labels[k] = vals_cat[i] 

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
#[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)