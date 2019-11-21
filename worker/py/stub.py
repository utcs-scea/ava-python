# vgg_preprocessing constants
class vgg_constants:

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    _RESIZE_SIDE_MIN = 256
    _RESIZE_SIDE_MAX = 512

class vgg_preprocessing:

    @staticmethod
    def _crop(image, offset_height, offset_width, crop_height, crop_width):
      """Crops the given image using the provided offsets and sizes.

      Note that the method doesn't assume we know the input image size but it does
      assume we know the input image rank.

      Args:
        image: an image of shape [height, width, channels].
        offset_height: a scalar tensor indicating the height offset.
        offset_width: a scalar tensor indicating the width offset.
        crop_height: the height of the cropped image.
        crop_width: the width of the cropped image.

      Returns:
        the cropped (and resized) image.

      Raises:
        InvalidArgumentError: if the rank is not 3 or if the image dimensions are
          less than the crop size.
      """
      original_shape = tf.shape(image)

      rank_assertion = tf.Assert(
          tf.equal(tf.rank(image), 3),
          ['Rank of image must be equal to 3.'])
      with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

      size_assertion = tf.Assert(
          tf.logical_and(
              tf.greater_equal(original_shape[0], crop_height),
              tf.greater_equal(original_shape[1], crop_width)),
          ['Crop size greater than the image size.'])

      offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

      # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
      # define the crop size.
      with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
      return tf.reshape(image, cropped_shape)


    @staticmethod
    def _random_crop(image_list, crop_height, crop_width):
      """Crops the given list of images.

      The function applies the same crop to each image in the list. This can be
      effectively applied when there are multiple image inputs of the same
      dimension such as:

        image, depths, normals = _random_crop([image, depths, normals], 120, 150)

      Args:
        image_list: a list of image tensors of the same dimension but possibly
          varying channel.
        crop_height: the new height.
        crop_width: the new width.

      Returns:
        the image_list with cropped images.

      Raises:
        ValueError: if there are multiple image inputs provided with different size
          or the images are smaller than the crop dimensions.
      """
      if not image_list:
        raise ValueError('Empty image_list.')

      # Compute the rank assertions.
      rank_assertions = []
      for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

      with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
      image_height = image_shape[0]
      image_width = image_shape[1]
      crop_size_assert = tf.Assert(
          tf.logical_and(
              tf.greater_equal(image_height, crop_height),
              tf.greater_equal(image_width, crop_width)),
          ['Crop size greater than the image size.'])

      asserts = [rank_assertions[0], crop_size_assert]

      for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
          shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

      # Create a random bounding box.
      #
      # Use tf.random_uniform and not numpy.random.rand as doing the former would
      # generate random numbers at graph eval time, unlike the latter which
      # generates random numbers at graph definition time.
      with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
      with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
      offset_height = tf.random_uniform(
          [], maxval=max_offset_height, dtype=tf.int32)
      offset_width = tf.random_uniform(
          [], maxval=max_offset_width, dtype=tf.int32)

      return [vgg_preprocessing._crop(image, offset_height, offset_width,
                    crop_height, crop_width) for image in image_list]


    @staticmethod
    def _central_crop(image_list, crop_height, crop_width):
      """Performs central crops of the given image list.

      Args:
        image_list: a list of image tensors of the same dimension but possibly
          varying channel.
        crop_height: the height of the image following the crop.
        crop_width: the width of the image following the crop.

      Returns:
        the list of cropped images.
      """
      outputs = []
      for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(vgg_preprocessing._crop(image, offset_height, offset_width,
                             crop_height, crop_width))
      return outputs


    @staticmethod
    def _mean_image_subtraction(image, means):
      """Subtracts the given means from each image channel.

      For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)

      Note that the rank of `image` must be known.

      Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.

      Returns:
        the centered image.

      Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
          than three or if the number of channels in `image` doesn't match the
          number of values in `means`.
      """
      if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
      num_channels = image.get_shape().as_list()[-1]
      if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

      channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
      for i in range(num_channels):
        channels[i] -= means[i]
      return tf.concat(axis=2, values=channels)


    @staticmethod
    def _smallest_size_at_least(height, width, smallest_side):
      """Computes new shape with the smallest side equal to `smallest_side`.

      Computes new shape with the smallest side equal to `smallest_side` while
      preserving the original aspect ratio.

      Args:
        height: an int32 scalar tensor indicating the current height.
        width: an int32 scalar tensor indicating the current width.
        smallest_side: A python integer or scalar `Tensor` indicating the size of
          the smallest side after resize.

      Returns:
        new_height: an int32 scalar tensor indicating the new height.
        new_width: and int32 scalar tensor indicating the new width.
      """
      smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

      height = tf.to_float(height)
      width = tf.to_float(width)
      smallest_side = tf.to_float(smallest_side)

      scale = tf.cond(tf.greater(height, width),
                      lambda: smallest_side / width,
                      lambda: smallest_side / height)
      new_height = tf.to_int32(height * scale)
      new_width = tf.to_int32(width * scale)
      return new_height, new_width


    @staticmethod
    def _aspect_preserving_resize(image, smallest_side):
      """Resize images preserving the original aspect ratio.

      Args:
        image: A 3-D image `Tensor`.
        smallest_side: A python integer or scalar `Tensor` indicating the size of
          the smallest side after resize.

      Returns:
        resized_image: A 3-D tensor containing the resized image.
      """
      smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

      shape = tf.shape(image)
      height = shape[0]
      width = shape[1]
      new_height, new_width = vgg_preprocessing._smallest_size_at_least(height, width, smallest_side)
      image = tf.expand_dims(image, 0)
      resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                               align_corners=False)
      resized_image = tf.squeeze(resized_image)
      resized_image.set_shape([None, None, 3])
      return resized_image


    @staticmethod
    def preprocess_for_train(image,
                             output_height,
                             output_width,
                             resize_side_min=vgg_constants._RESIZE_SIDE_MIN,
                             resize_side_max=vgg_constants._RESIZE_SIDE_MAX):
      """Preprocesses the given image for training.

      Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

      Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
          aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
          aspect-preserving resizing.

      Returns:
        A preprocessed image.
      """
      resize_side = tf.random_uniform(
          [], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)

      image = vgg_preprocessing._aspect_preserving_resize(image, resize_side)
      image = vgg_preprocessing._random_crop([image], output_height, output_width)[0]
      image.set_shape([output_height, output_width, 3])
      image = tf.to_float(image)
      image = tf.image.random_flip_left_right(image)
      return vgg_preprocessing._mean_image_subtraction(image,
              [vgg_constants._R_MEAN, vgg_constants._G_MEAN,
                  vgg_constants._B_MEAN])

    @staticmethod
    def preprocess_for_eval(image, output_height, output_width, resize_side):
      """Preprocesses the given image for evaluation.

      Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side: The smallest side of the image for aspect-preserving resizing.

      Returns:
        A preprocessed image.
      """
      image = vgg_preprocessing._aspect_preserving_resize(image, resize_side)
      image = vgg_preprocessing._central_crop([image], output_height, output_width)[0]
      image.set_shape([output_height, output_width, 3])
      image = tf.to_float(image)
      return vgg_preprocessing._mean_image_subtraction(image,
              [vgg_constants._R_MEAN, vgg_constants._G_MEAN,
                  vgg_constants._B_MEAN])

    @staticmethod
    def preprocess_image(image, output_height, output_width, is_training=False,
                         resize_side_min=vgg_constants._RESIZE_SIDE_MIN,
                         resize_side_max=vgg_constants._RESIZE_SIDE_MAX):
      """Preprocesses the given image.

      Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        is_training: `True` if we're preprocessing the image for training and
          `False` otherwise.
        resize_side_min: The lower bound for the smallest side of the image for
          aspect-preserving resizing. If `is_training` is `False`, then this value
          is used for rescaling.
        resize_side_max: The upper bound for the smallest side of the image for
          aspect-preserving resizing. If `is_training` is `False`, this value is
          ignored. Otherwise, the resize side is sampled from
            [resize_size_min, resize_size_max].

      Returns:
        A preprocessed image.
      """
      if is_training:
        image = vgg_preprocessing.preprocess_for_train(image, output_height, output_width,
                                     resize_side_min, resize_side_max)
      else:
        image = vgg_preprocessing.preprocess_for_eval(image, output_height, output_width,
                                    resize_side_min)
      # Scale to (-1,1). TODO(currutia): check whether this is actually needed
      image = tf.multiply(image, 1. / 128.)
      return image


class inception_preprocessing:

    @staticmethod
    def apply_with_random_selector(x, func, num_cases):
      """Computes func(x, sel), with sel sampled from [0...num_cases-1].

      Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

      Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
      """
      sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
      # Pass the real x only to one of the func calls.
      return control_flow_ops.merge([
          func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
          for case in range(num_cases)])[0]

    @staticmethod
    def distort_color_fast(image, scope=None):
      """Distort the color of a Tensor image.

      Distort brightness and chroma values of input image

      Args:
        image: 3-D Tensor containing single image in [0, 1].
        scope: Optional scope for name_scope.
      Returns:
        3-D Tensor color-distorted image on range [0, 1]
      """
      with tf.name_scope(scope, 'distort_color', [image]):
        br_delta = random_ops.random_uniform([], -32./255., 32./255., seed=None)
        cb_factor = random_ops.random_uniform(
            [], -FLAGS.cb_distortion_range, FLAGS.cb_distortion_range, seed=None)
        cr_factor = random_ops.random_uniform(
            [], -FLAGS.cr_distortion_range, FLAGS.cr_distortion_range, seed=None)

        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        red_offset = 1.402 * cr_factor + br_delta
        green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
        blue_offset = 1.772 * cb_factor + br_delta
        channels[0] += red_offset
        channels[1] += green_offset
        channels[2] += blue_offset
        image = tf.concat(axis=2, values=channels)
        image = tf.clip_by_value(image, 0., 1.)

        return image

    @staticmethod
    def distorted_bounding_box_crop(image,
                                    bbox,
                                    min_object_covered=0.1,
                                    aspect_ratio_range=(3./4., 4./3.),
                                    area_range=(0.05, 1.0),
                                    max_attempts=100,
                                    scope=None):
      """Generates cropped_image using a one of the bboxes randomly distorted.

      See `tf.image.sample_distorted_bounding_box` for more documentation.

      Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged
          as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
          image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding box
          supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
          image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
        scope: Optional scope for name_scope.
      Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
      """
      with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox

    @staticmethod
    def preprocess_for_train(image, height, width, bbox,
                             fast_mode=True,
                             scope=None,
                             add_image_summaries=True):
      """Distort one image for training a network.

      Distorting images provides a useful technique for augmenting the data
      set during training in order to make the network invariant to aspects
      of the image that do not effect the label.

      Additionally it would create image_summaries to display the different
      transformations applied to the image.

      Args:
        image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
          [0, 1], otherwise it would converted to tf.float32 assuming that the range
          is [0, MAX], where MAX is largest positive representable number for
          int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
        height: integer
        width: integer
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged
          as [ymin, xmin, ymax, xmax].
        fast_mode: Optional boolean, if True avoids slower transformations (i.e.
          bi-cubic resizing, random_hue or random_contrast).
        scope: Optional scope for name_scope.
        add_image_summaries: Enable image summaries.
      Returns:
        3-D float Tensor of distorted image used for training with range [-1, 1].
      """
      with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
          bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                             dtype=tf.float32,
                             shape=[1, 1, 4])
        if image.dtype != tf.float32:
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if add_image_summaries:
          # Each bounding box has shape [1, num_boxes, box coords] and
          # the coordinates are ordered [ymin, xmin, ymax, xmax].
          image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                        bbox)
          tf.summary.image('image_with_bounding_boxes', image_with_box)

        distorted_image, distorted_bbox = inception_preprocessing.distorted_bounding_box_crop(image, bbox)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        if add_image_summaries:
          image_with_distorted_box = tf.image.draw_bounding_boxes(
              tf.expand_dims(image, 0), distorted_bbox)
          tf.summary.image('images_with_distorted_bounding_box',
                           image_with_distorted_box)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = inception_preprocessing.apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [height, width], method),
            num_cases=num_resize_cases)

        if add_image_summaries:
          tf.summary.image('cropped_resized_image',
                           tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors. There are 1 or 4 ways to do it.
        if FLAGS.use_fast_color_distort:
          distorted_image = inception_preprocessing.distort_color_fast(distorted_image)
        else:
          num_distort_cases = 1 if fast_mode else 4
          distorted_image = inception_preprocessing.apply_with_random_selector(
              distorted_image,
              lambda x, ordering: distort_color(x, ordering, fast_mode),
              num_cases=num_distort_cases)

        if add_image_summaries:
          tf.summary.image('final_distorted_image',
                           tf.expand_dims(distorted_image, 0))
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image


    @staticmethod
    def preprocess_for_eval(image, height, width,
                            central_fraction=0.875, scope=None):
      """Prepare one image for evaluation.

      If height and width are specified it would output an image with that size by
      applying resize_bilinear.

      If central_fraction is specified it would crop the central fraction of the
      input image.

      Args:
        image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
          [0, 1], otherwise it would converted to tf.float32 assuming that the range
          is [0, MAX], where MAX is largest positive representable number for
          int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
        height: integer
        width: integer
        central_fraction: Optional Float, fraction of the image to crop.
        scope: Optional scope for name_scope.
      Returns:
        3-D float Tensor of prepared image.
      """
      with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
          image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
          # Resize the image to the specified height and width.
          image = tf.expand_dims(image, 0)
          image = tf.image.resize_bilinear(image, [height, width],
                                           align_corners=False)
          image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        image.set_shape([height, width, 3])
        return image

    @staticmethod
    def preprocess_image(image, output_height, output_width,
                         is_training=False,
                         bbox=None,
                         fast_mode=True,
                         add_image_summaries=False):
      """Pre-process one image for training or evaluation.

      Args:
        image: 3-D Tensor [height, width, channels] with the image. If dtype is
          tf.float32 then the range should be [0, 1], otherwise it would converted
          to tf.float32 assuming that the range is [0, MAX], where MAX is largest
          positive representable number for int(8/16/32) data type (see
          `tf.image.convert_image_dtype` for details).
        output_height: integer, image expected height.
        output_width: integer, image expected width.
        is_training: Boolean. If true it would transform an image for train,
          otherwise it would transform it for evaluation.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged as
          [ymin, xmin, ymax, xmax].
        fast_mode: Optional boolean, if True avoids slower transformations.
        add_image_summaries: Enable image summaries.

      Returns:
        3-D float Tensor containing an appropriately scaled image

      Raises:
        ValueError: if user does not provide bounding box
      """
      if is_training:
        return inception_preprocessing.preprocess_for_train(image,
                                    output_height, output_width, bbox,
                                    fast_mode,
                                    add_image_summaries=add_image_summaries)
      else:
        return inception_preprocessing.preprocess_for_eval(image,
                                    output_height, output_width)


