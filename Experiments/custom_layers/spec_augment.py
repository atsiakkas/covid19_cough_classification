import tensorflow as tf

class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, time_warping, frequency_masking, frequency_mask_num,
                 time_masking, time_mask_num, **kwargs):
        super(SpecAugment, self).__init__(**kwargs)
        self.time_warping = time_warping
        self.frequency_masking = frequency_masking
        self.frequency_mask_num = frequency_mask_num
        self.time_masking = time_masking
        self.time_mask_num = time_mask_num

    # def build(self, input_shape):
    #     self.non_trainable_weights.append(self.mel_filterbank)
    #     super(LogMelSpectrogram, self).build(input_shape)

    def call(self, mel_spectrograms, training=None):
        '''Forward pass.

        Parameters
        ----------
        spectrograms : tf.Tensor, shape = (None, time, freq, ch)
            A Batch of spectrograms.

        Returns
        -------
        Spec Augment spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of spec augmented spectrograms
        '''

        if training:
            # commenting out until we have support for tensorflow.addons
            # def sparse_warp(mel_spectrogram, time_warping=self.time_warping):
            #     """Spec augmentation Calculation Function.
            #     # Arguments:
            #     mel_spectrogram(numpy array): audio file path of you want to warping and masking.
            #     time_warping(float): Augmentation parameter, "time warp parameter W".
            #         If none, default = 80 for LibriSpeech.
            #     # Returns
            #     mel_spectrogram(numpy array): warped and masked mel spectrogram.
            #     """

            #     fbank_size = tf.shape(mel_spectrogram)
            #     n, v = fbank_size[1], fbank_size[2]

            #     # Step 1 : Time warping
            #     # Image warping control point setting.
            #     # Source
            #     pt = tf.random.uniform(
            #         [], time_warping, n - time_warping,
            #         tf.int32)  # random point along the time axis
            #     src_ctr_pt_freq = tf.range(v //
            #                                2)  # control points on freq-axis
            #     src_ctr_pt_time = tf.ones_like(
            #         src_ctr_pt_freq) * pt  # control points on time-axis
            #     src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
            #     src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

            #     # Destination
            #     w = tf.random.uniform([], -time_warping, time_warping,
            #                           tf.int32)  # distance
            #     dest_ctr_pt_freq = src_ctr_pt_freq
            #     dest_ctr_pt_time = src_ctr_pt_time + w
            #     dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq),
            #                             -1)
            #     dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

            #     # warp
            #     source_control_point_locations = tf.expand_dims(
            #         src_ctr_pts, 0)  # (1, v//2, 2)
            #     dest_control_point_locations = tf.expand_dims(
            #         dest_ctr_pts, 0)  # (1, v//2, 2)

            #     warped_image, _ = sparse_image_warp(
            #         mel_spectrogram, source_control_point_locations,
            #         dest_control_point_locations)
            #     return warped_image

            def frequency_masking(mel_spectrogram,
                                  v,
                                  frequency_masking=self.frequency_masking,
                                  frequency_mask_num=self.frequency_mask_num):
                """Spec augmentation Calculation Function.
                # Arguments:
                mel_spectrogram(numpy array): audio file path of you want to warping and masking.
                frequency_masking(float): Augmentation parameter, "frequency mask parameter F"
                    If none, default = 100 for LibriSpeech.
                frequency_mask_num(float): number of frequency masking lines, "m_F".
                    If none, default = 1 for LibriSpeech.
                # Returns
                mel_spectrogram(numpy array): frequency masked mel spectrogram.
                    """
                # Step 2 : Frequency masking
                # These two lines are not required, they can be derived from the params
                fbank_size = tf.shape(mel_spectrogram)
                n, v = fbank_size[1], fbank_size[2]

                for i in range(frequency_mask_num):
                    f = tf.random.uniform([],
                                          minval=0,
                                          maxval=frequency_masking,
                                          dtype=tf.int32)
                    v = tf.cast(v, dtype=tf.int32)
                    f0 = tf.random.uniform([],
                                           minval=0,
                                           maxval=v - f,
                                           dtype=tf.int32)

                    # warped_mel_spectrogram[f0:f0 + f, :] = 0
                    mask = tf.concat((
                        tf.ones(shape=(1, n, v - f0 - f, 1)),
                        tf.zeros(shape=(1, n, f, 1)),
                        tf.ones(shape=(1, n, f0, 1)),
                    ), 2)
                    mel_spectrogram = mel_spectrogram * mask
                return tf.cast(mel_spectrogram, dtype=tf.float32)

            def time_masking(mel_spectrogram,
                             tau,
                             time_masking=self.time_masking,
                             time_mask_num=self.time_mask_num):
                # time_masking was 100
                """Spec augmentation Calculation Function.
                # Arguments:
                mel_spectrogram(numpy array): audio file path of you want to warping and masking.
                time_masking(float): Augmentation parameter, "time mask parameter T"
                    If none, default = 27 for LibriSpeech.
                time_mask_num(float): number of time masking lines, "m_T".
                    If none, default = 1 for LibriSpeech.
                # Returns
                mel_spectrogram(numpy array): warped and masked mel spectrogram.
                """
                fbank_size = tf.shape(mel_spectrogram)
                n, v = fbank_size[1], fbank_size[2]

                # Step 3 : Time masking
                for i in range(time_mask_num):
                    t = tf.random.uniform([],
                                          minval=0,
                                          maxval=time_masking,
                                          dtype=tf.int32)
                    t0 = tf.random.uniform([],
                                           minval=0,
                                           maxval=tau - t,
                                           dtype=tf.int32)

                    # mel_spectrogram[:, t0:t0 + t] = 0
                    mask = tf.concat((
                        tf.ones(shape=(1, n - t0 - t, v, 1)),
                        tf.zeros(shape=(1, t, v, 1)),
                        tf.ones(shape=(1, t0, v, 1)),
                    ), 1)
                    mel_spectrogram = mel_spectrogram * mask
                return tf.cast(mel_spectrogram, dtype=tf.float32)

            def spec_augment(mel_spectrogram):

                v = mel_spectrogram.shape[0]
                tau = mel_spectrogram.shape[1]

                # commenting out until we have support for tensorflow.addons
                # warped_mel_spectrogram = sparse_warp(mel_spectrogram)

                # warped_frequency_spectrogram = frequency_masking(
                #     warped_mel_spectrogram, v=v)

                # temp line:
                warped_frequency_spectrogram = frequency_masking(
                    mel_spectrogram, v=v)

                warped_frequency_time_spectrogram = time_masking(
                    warped_frequency_spectrogram, tau=tau)

                return warped_frequency_time_spectrogram

            return spec_augment(mel_spectrograms)
        else:
            return mel_spectrograms

    def get_config(self):
        config = {
            'time_warping': self.time_warping,
            'frequency_masking': self.frequency_masking,
            'frequency_mask_num': self.frequency_mask_num,
            'time_masking': self.time_masking,
            'time_mask_num': self.time_mask_num
        }
        config.update(super(SpecAugment, self).get_config())

        return config
