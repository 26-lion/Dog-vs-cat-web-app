from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(directory="train2/train", target_size=(150, 150), class_mode="binary", batch_size=100)
validate_generator = train_datagen.flow_from_directory(directory="train2/validation", target_size=(150, 150), class_mode="binary", batch_size=100)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", padding="valid", input_shape=(150, 150, 3)))
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation="relu", padding="valid"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (2, 2), activation="relu", padding="valid"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
print(model.summary())
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
model.fit_generator(train_generator, epochs=10, validation_data=validate_generator)
model.save("Classifier")
print("model saved")



