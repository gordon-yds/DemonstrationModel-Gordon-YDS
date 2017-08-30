def get_model(trainer):
    """Returns the model (not compiled, not trained)

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """

    import keras
    import keras.models
    import keras.layers
    import keras.layers.convolutional
    import keras.layers.core

    input = keras.layers.Input(shape=(128, 128, 3), name="Input")
    conv1 = keras.layers.convolutional.Convolution2D(20, 3, 3, border_mode='valid', name="conv1", activation='relu',
                                                     init='glorot_uniform')(input)
    pool1 = keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), name="pool1")(conv1)
    conv2 = keras.layers.convolutional.Convolution2D(20, 3, 3, border_mode='valid', name="conv2", activation='relu',
                                                     init='glorot_uniform')(pool1)
    pool2 = keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), name="pool2")(conv2)
    conv3 = keras.layers.convolutional.Convolution2D(25, 3, 3, border_mode='valid', name="conv3", activation='relu',
                                                     init='glorot_uniform')(pool2)
    pool3 = keras.layers.core.Flatten(name="pool3")(conv3)
    dense = keras.layers.core.Dense(30, name="dense", activation='relu', init='glorot_uniform')(pool3)
    dense = keras.layers.core.Dropout(0.5, name="dense_dropout")(dense)
    out = keras.layers.core.Dense(2, name="out", activation='softmax', init='glorot_uniform')(dense)

    return keras.models.Model([input], [out])


def compile(trainer, model, loss, optimizer):
    """Compiles the given model (from get_model) with given loss (from get_loss) and optimizer (from get_optimizer)

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


def get_training_data(trainer, dataset):
    """Returns the training and validation data from dataset. Since dataset is living in its own domain
    it could be necessary to transform the given dataset to match the model's input layer. (e.g. convolution vs dense)

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer

    dataset: dict
        Contains for each used dataset a key => dictionary, where key is the dataset id and the dict is returned from the Dataset "get_data" method.
        Their keys: 'X_train', 'Y_train', 'X_test', 'Y_test'.
        Example data['mnistExample']['X_train'] for a input layer that has 'mnistExample' as chosen dataset.
    """

    return {
        'x': {'Input': dataset['gordonhopgood/dataset/cats-dogs']['X_train']},
        'y': {'out': dataset['gordonhopgood/dataset/cats-dogs']['Y_train']}
    }


def get_validation_data(trainer, dataset):
    """Returns the training and validation data from dataset.

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer

    dataset: dict
        Contains for each used dataset a key => dictionary, where key is the dataset id and the dict is returned from the Dataset "get_data" method.
        Their keys: 'X_train', 'Y_train', 'X_test', 'Y_test'.
        Example data['mnistExample']['X_test'] for a input layer that has 'mnistExample' as chosen dataset.
    """

    return {
        'x': {'Input': dataset['gordonhopgood/dataset/cats-dogs']['X_test']},
        'y': {'out': dataset['gordonhopgood/dataset/cats-dogs']['Y_test']}
    }


def get_optimizer(trainer):
    """Returns the optimizer

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """

    from aetros.keras import optimizer_factory

    return optimizer_factory(trainer.job_backend.get_parameter('keras_optimizer'))


def get_loss(trainer):
    """Returns the optimizer

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """

    loss = {'out': 'categorical_crossentropy'}
    return loss


def train(trainer, model, training_data, validation_data):
    """Returns the model (not build, not trained)

    Parameters
    ----------
    trainer : aetros.Trainer.Trainer
    """
    nb_epoch = trainer.settings['epochs']
    batch_size = trainer.settings['batchSize']

    if trainer.has_generator(training_data['x']):
        model.fit_generator(
            trainer.get_first_generator(training_data['x']),
            samples_per_epoch=trainer.samples_per_epoch,
            nb_val_samples=trainer.nb_val_samples,

            nb_epoch=nb_epoch,
            verbose=0,
            validation_data=trainer.get_first_generator(validation_data['x']),
            callbacks=trainer.callbacks
        )
    else:
        model.fit(
            training_data['x'],
            training_data['y'],
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            verbose=0,
            validation_data=(validation_data['x'], validation_data['y']),
            callbacks=trainer.callbacks
        )

