import tensorflow as tf


class Dataset:
    def __init__(
        self,
        train_path,
        test_path,
        img_height,
        img_width,
        buffer_size,
        batch_size,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.train = self.create_dataset(
            train_path,
            self._load_image_train,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        self.test = self.create_dataset(
            test_path,
            self._load_image_test,
        )

    def create_dataset(self, path, map_func, num_parallel_calls=None):
        dataset = tf.data.Dataset.list_files(path)
        if num_parallel_calls:
            dataset = dataset.map(map_func, num_parallel_calls=num_parallel_calls)
        else:
            dataset = dataset.map(map_func)
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _load(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        w = tf.shape(image)[1]

        w = w // 2
        real_image = image[:, :w, :]
        input_image = image[:, w:, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def _resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(
            input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        real_image = tf.image.resize(
            real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        return input_image, real_image

    def _random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, self.img_height, self.img_width, 3]
        )

        return cropped_image[0], cropped_image[1]

    def _normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    @tf.function()
    def _random_jitter(self, input_image, real_image):
        input_image, real_image = self._resize(input_image, real_image, 286, 286)
        input_image, real_image = self._random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def _load_image_train(self, image_file):
        input_image, real_image = self._load(image_file)
        input_image, real_image = self._random_jitter(input_image, real_image)
        input_image, real_image = self._normalize(input_image, real_image)

        return input_image, real_image

    def _load_image_test(self, image_file):
        input_image, real_image = self._load(image_file)
        input_image, real_image = self._resize(
            input_image, real_image, self.img_height, self.img_width
        )
        input_image, real_image = self._normalize(input_image, real_image)

        return input_image, real_image
