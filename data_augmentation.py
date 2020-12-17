import tensorflow as tf
import os


def main(clear=False, clear_only=False, n_new_images=32):
    # create necessary folders
    dataset_path = "set"
    for directory in os.listdir(dataset_path):
        class_path = dataset_path + "/" + directory
        if os.path.isdir(class_path) and not os.path.isdir(class_path + "/original"):
            os.mkdir(class_path + "/original")
            os.mkdir(class_path + "/augmented")
            for file in os.listdir(class_path):
                if os.path.isfile(class_path + "/" + file):
                    os.rename(class_path + "/" + file, class_path + "/original/" + file)
        elif clear or clear_only:
            for augmented in os.listdir(class_path + "/augmented"):
                os.unlink(class_path + "/augmented/" + augmented)
    if clear_only:
        return
        # for each folder augment the set
    for directory in os.listdir(dataset_path):
        if os.path.isdir(dataset_path + "/" + directory):
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='nearest')
            train_generator = train_datagen.flow_from_directory(
                directory='./set/' + directory,
                class_mode='categorical',
                batch_size=1,
                save_to_dir='./set/' + directory + "/augmented",
                save_format="png",
            )
            for i in range(n_new_images):
                train_generator.next()


if __name__ == "__main__":
    main(clear=True, n_new_images=20)
