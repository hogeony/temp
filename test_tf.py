# tensorflow-gpu 2.0.0b1 (gpu, CUDA 10)
import time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import *

class Model(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = Sequential([
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(filters=32,kernel_size=(5,5),padding='same', activation='relu'),
            MaxPool2D(pool_size=(2,2)),
            Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=1024, activation='relu'),
            Dropout(rate=0.5),
            Dense(units=10, activation='softmax'),])

    def call(self, inputs):
        return self.layer(inputs)

(x_train, y_train),(x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
x_train, x_test = x_train.reshape((60000, 28, 28, 1)), x_test.reshape((10000, 28, 28, 1))

c = Model()
optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
c.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(), metrics=[metrics.SparseCategoricalAccuracy()])

time_start = time.time()
c.fit(x_train, y_train, epochs=3, batch_size=100, shuffle=True)
time_end = time.time()
print(time_end-time_start)
# 0.9900
# 15.218996286392212

time_start = time.time()
c.evaluate(x_test, y_test)
time_end = time.time()
print(time_end-time_start)
# 0.9909
# 0.7690010070800781

c.layer.summary()
# 3,274,634
