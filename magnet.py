from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import cv2
import time

from glob import glob
from scipy.signal import firwin, butter
from functools import partial
from tqdm import tqdm, trange
from subprocess import call

from modules import L1_loss
from modules import res_encoder, res_decoder, res_manipulator
from modules import residual_block, conv2d
from utils import load_train_data, mkdir, imread, save_images
from preprocessor import preprocess_image, preproc_color
from data_loader import read_and_decode_3frames

# Change here if you use ffmpeg.
DEFAULT_VIDEO_CONVERTER = 'avconv'


class MagNet3Frames(object):

    def __init__(self, sess, name, arch_config):
        self.sess = sess
        self.exp_name = name
        self.is_graph_built = False
        self.n_channels = arch_config["n_channels"]
        self.arch_config = arch_config
        self.encoder_dims = arch_config["ynet_3frames"]["enc_dims"]
        self.num_enc_resblk = arch_config["ynet_3frames"]["num_enc_resblk"]
        self.num_man_resblk = arch_config["ynet_3frames"]["num_man_resblk"]
        self.num_man_conv = arch_config["ynet_3frames"]["num_man_conv"]
        self.num_man_aft_conv = arch_config["ynet_3frames"]["num_man_aft_conv"]
        self.num_dec_resblk = arch_config["ynet_3frames"]["num_dec_resblk"]
        self.num_texture_resblk = \
            arch_config["ynet_3frames"]["num_texture_resblk"]
        self.texture_dims = arch_config["ynet_3frames"]["texture_dims"]
        self.texture_downsample = \
            arch_config["ynet_3frames"]["texture_downsample"]
        self.use_texture_conv = arch_config["ynet_3frames"]["use_texture_conv"]
        self.shape_dims = arch_config["ynet_3frames"]["shape_dims"]
        self.num_shape_resblk = \
            arch_config["ynet_3frames"]["num_shape_resblk"]
        self.use_shape_conv = arch_config["ynet_3frames"]["use_shape_conv"]
        self.decoder_dims = self.texture_dims + self.shape_dims
        self.probe_pt = {}
        self.manipulator = partial(res_manipulator,
                                   layer_dims=self.encoder_dims,
                                   num_resblk=self.num_man_resblk,
                                   num_conv=self.num_man_conv,
                                   num_aft_conv=self.num_man_aft_conv,
                                   probe_pt=self.probe_pt)

    def _encoder(self, image):
        enc = res_encoder(image,
                          layer_dims=self.encoder_dims,
                          num_resblk=self.num_enc_resblk)

        texture_enc = enc
        shape_enc = enc
        # first convolution on common encoding
        if self.use_texture_conv:
            stride = 2 if self.texture_downsample else 1
            texture_enc = tf.nn.relu(conv2d(texture_enc, self.texture_dims,
                                            3, stride,
                                            name='enc_texture_conv'))
        else:
            assert self.texture_dims == self.encoder_dims, \
                "Texture dim ({}) must match encoder dim ({}) " \
                "if texture_conv is not used.".format(self.texture_dims,
                                                      self.encoder_dims)
            assert not self.texture_downsample, \
                "Must use texture_conv if texture_downsample."
        if self.use_shape_conv:
            shape_enc = tf.nn.relu(conv2d(shape_enc, self.shape_dims,
                                          3, 1, name='enc_shape_conv'))
        else:
            assert self.shape_dims == self.encoder_dims, \
                "Shape dim ({}) must match encoder dim ({}) " \
                "if shape_conv is not used.".format(self.shape_dims,
                                                    self.encoder_dims)

        for i in range(self.num_texture_resblk):
            name = 'texture_enc_{}'.format(i)
            if i == 0:
                # for backward compatibility
                name = 'texture_enc'
            texture_enc = residual_block(texture_enc, self.texture_dims, 3, 1,
                                         name)
        for i in range(self.num_shape_resblk):
            name = 'shape_enc_{}'.format(i)
            if i == 0:
                # for backward compatibility
                name = 'shape_enc'
            shape_enc = residual_block(shape_enc, self.shape_dims,
                                       3, 1, name)
        return texture_enc, shape_enc

    def _decoder(self, texture_enc, shape_enc):
        if self.texture_downsample:
            texture_enc = tf.image.resize_nearest_neighbor(
                            texture_enc,
                            tf.shape(texture_enc)[1:3] \
                            * 2)
            texture_enc = tf.pad(texture_enc, [[0, 0], [1, 1], [1, 1], [0, 0]],
                                 "REFLECT")
            texture_enc = tf.nn.relu(conv2d(texture_enc, self.texture_dims,
                                            3, 1, padding='VALID',
                                            name='texture_upsample'))

        enc = tf.concat([texture_enc, shape_enc], axis=3)
        # Needs double the channel because we concat the two encodings.
        return res_decoder(enc,
                           layer_dims=self.decoder_dims,
                           out_channels=self.n_channels,
                           num_resblk=self.num_dec_resblk)

    def image_transformer(self,
                          image_a,
                          image_b,
                          amplification_factor,
                          im_size,
                          options,
                          is_training,
                          reuse=False,
                          name='ynet_3frames'):
        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope('encoder'):
                self.texture_a, self.shape_a = self._encoder(image_a)
            with tf.variable_scope('encoder', reuse=True):
                self.texture_b, self.shape_b = self._encoder(image_b)
            with tf.variable_scope('manipulator'):
                self.out_shape_enc = self.manipulator(self.shape_a,
                                                      self.shape_b,
                                                      amplification_factor)
            with tf.variable_scope('decoder'):
                return self._decoder(self.texture_b, self.out_shape_enc)

    def save(self, checkpoint_dir, step):
        model_name = self.exp_name

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, loader=None):
        if not loader:
            loader = self.saver
        print(" [*] Reading checkpoint...")
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = ckpt.model_checkpoint_path
            else:
                ckpt_name = None
        else:
            # load from file
            ckpt_name = checkpoint_dir
        if ckpt_name:
            loader.restore(self.sess, ckpt_name)
            print('Loaded from ckpt: ' + ckpt_name)
            self.ckpt_name = ckpt_name
            return True
        else:
            return False

    def _build_feed_model(self):
        self.test_input = tf.placeholder(tf.float32,
                                         [None, None, None,
                                             self.n_channels * 3],
                                         name='test_AB_and_output')
        self.test_amplification_factor = tf.placeholder(tf.float32,
                                                        [None],
                                                        name='amplification_factor')
        self.test_image_a = self.test_input[:, :, :, :self.n_channels]
        self.test_image_b = self.test_input[:, :, :, self.n_channels:(2 * self.n_channels)]
        self.test_amplified_frame = self.test_input[:, :, :, (2*self.n_channels):(3 * self.n_channels)]
        self.test_output = self.image_transformer(
                               self.test_image_a,
                               self.test_image_b,
                               self.test_amplification_factor,
                               [self.image_height, self.image_width],
                               self.arch_config,
                               False,
                               False)
        self.test_output = tf.clip_by_value(self.test_output, -1.0, 1.0)
        self.saver = tf.train.Saver()
        self.is_graph_built = True

    def setup_for_inference(self, checkpoint_dir, image_width, image_height):
        """Setup model for inference.

        Build computation graph, initialize variables, and load checkpoint.
        """
        self.image_width = image_width
        self.image_height = image_height
        # Figure out image dimension
        self._build_feed_model()
        ginit_op = tf.global_variables_initializer()
        linit_op = tf.local_variables_initializer()
        self.sess.run([ginit_op, linit_op])

        if self.load(checkpoint_dir):
            print("[*] Load Success")
        else:
            raise RuntimeError('MagNet: Failed to load checkpoint file.')
        self.is_graph_built = True

    def inference(self, frameA, frameB, amplification_factor):
        """Run Magnification on two frames.

        Args:
            frameA: path to first frame
            frameB: path to second frame
            amplification_factor: float for amplification factor
        """
        in_frames = [load_train_data([frameA, frameB, frameB],
                     gray_scale=self.n_channels==1, is_testing=True)]
        in_frames = np.array(in_frames).astype(np.float32)

        out_amp = self.sess.run(self.test_output,
                                feed_dict={self.test_input: in_frames,
                                           self.test_amplification_factor:
                                           [amplification_factor]})
        return out_amp

    def run(self,
            checkpoint_dir,
            vid_dir,
            frame_ext,
            out_dir,
            amplification_factor,
            velocity_mag=False):
        """Magnify a video in the two-frames mode.

        Args:
            checkpoint_dir: checkpoint directory.
            vid_dir: directory containing video frames videos are processed
                in sorted order.
            out_dir: directory to place output frames and resulting video.
            amplification_factor: the amplification factor,
                with 0 being no change.
            velocity_mag: if True, process video in Dynamic mode.
        """
        vid_name = os.path.basename(out_dir)
        # make folder
        mkdir(out_dir)
        vid_frames = sorted(glob(os.path.join(vid_dir, '*.' + frame_ext)))
        first_frame = vid_frames[0]
        im = imread(first_frame)
        image_height, image_width = im.shape
        if not self.is_graph_built:
            self.setup_for_inference(checkpoint_dir, image_width, image_height)
        try:
            i = int(self.ckpt_name.split('-')[-1])
            print("Iteration number is {:d}".format(i))
            vid_name = vid_name + '_' + str(i)
        except:
            print("Cannot get iteration number")
        if velocity_mag:
            print("Running in Dynamic mode")

        prev_frame = first_frame
        desc = vid_name if len(vid_name) < 10 else vid_name[:10]
        for frame in tqdm(vid_frames, desc=desc):
            file_name = os.path.basename(frame)
            out_amp = self.inference(prev_frame, frame, amplification_factor)

            im_path = os.path.join(out_dir, file_name)
            save_images(out_amp, [1, 1], im_path)
            if velocity_mag:
                prev_frame = frame

        # Try to combine it into a video
        call([DEFAULT_VIDEO_CONVERTER, '-y', '-f', 'image2', '-r', '30', '-i',
              os.path.join(out_dir, '%06d.png'), '-c:v', 'libx264',
              os.path.join(out_dir, vid_name + '.mp4')]
            )

    # Temporal Operations
    def _build_IIR_filtering_graphs(self):
        """
        Assume a_0 = 1
        """
        self.input_image = tf.placeholder(tf.float32,
                                          [1, self.image_height,
                                              self.image_width,
                                           self.n_channels],
                                          name='input_image')
        self.filtered_enc = tf.placeholder(tf.float32,
                                           [1, None, None,
                                            self.shape_dims],
                                           name='filtered_enc')
        self.out_texture_enc = tf.placeholder(tf.float32,
                                              [1, None, None,
                                               self.texture_dims],
                                              name='out_texture_enc')
        self.ref_shape_enc = tf.placeholder(tf.float32,
                                            [1, None, None,
                                             self.shape_dims],
                                            name='ref_shape_enc')
        self.amplification_factor = tf.placeholder(tf.float32, [None],
                                                   name='amplification_factor')
        with tf.variable_scope('ynet_3frames'):
            with tf.variable_scope('encoder'):
                self.texture_enc, self.shape_rep = \
                    self._encoder(self.input_image)
            with tf.variable_scope('manipulator'):
                # set encoder a to zero because we do temporal filtering
                # instead of taking the difference.
                self.out_shape_enc = self.manipulator(0.0,
                                                      self.filtered_enc,
                                                      self.amplification_factor)
                self.out_shape_enc += self.ref_shape_enc - self.filtered_enc
            with tf.variable_scope('decoder'):
                self.output_image = tf.clip_by_value(
                                        self._decoder(self.out_texture_enc,
                                                      self.out_shape_enc),
                                        -1.0, 1.0)

        self.saver = tf.train.Saver()

    def run_temporal(self,
                     checkpoint_dir,
                     vid_dir,
                     frame_ext,
                     out_dir,
                     amplification_factor,
                     fl, fh, fs,
                     n_filter_tap,
                     filter_type):
        """Magnify video with a temporal filter.

        Args:
            checkpoint_dir: checkpoint directory.
            vid_dir: directory containing video frames videos are processed
                in sorted order.
            out_dir: directory to place output frames and resulting video.
            amplification_factor: the amplification factor,
                with 0 being no change.
            fl: low cutoff frequency.
            fh: high cutoff frequency.
            fs: sampling rate of the video.
            n_filter_tap: number of filter tap to use.
            filter_type: Type of filter to use. Can be one of "fir",
                "butter", or "differenceOfIIR". For "differenceOfIIR",
                fl and fh specifies rl and rh coefficients as in Wadhwa et al.
        """

        nyq = fs / 2.0
        if filter_type == 'fir':
            filter_b = firwin(n_filter_tap, [fl, fh], nyq=nyq, pass_zero=False)
            filter_a = []
        elif filter_type == 'butter':
            filter_b, filter_a = butter(n_filter_tap, [fl/nyq, fh/nyq],
                                        btype='bandpass')
            filter_a = filter_a[1:]
        elif filter_type == 'differenceOfIIR':
            # This is a copy of what Neal did. Number of taps are ignored.
            # Treat fl and fh as rl and rh as in Wadhwa's code.
            # Write down the difference of difference equation in Fourier
            # domain to proof this:
            filter_b = [fh - fl, fl - fh]
            filter_a = [-1.0*(2.0 - fh - fl), (1.0 - fl) * (1.0 - fh)]
        else:
            raise ValueError('Filter type must be either '
                             '["fir", "butter", "differenceOfIIR"] got ' + \
                             filter_type)
        head, tail = os.path.split(out_dir)
        tail = tail + '_fl{}_fh{}_fs{}_n{}_{}'.format(fl, fh, fs,
                                                      n_filter_tap,
                                                      filter_type)
        out_dir = os.path.join(head, tail)
        vid_name = os.path.basename(out_dir)
        # make folder
        mkdir(out_dir)
        vid_frames = sorted(glob(os.path.join(vid_dir, '*.' + frame_ext)))
        first_frame = vid_frames[0]
        im = imread(first_frame)
        image_height, image_width = im.shape
        if not self.is_graph_built:
            self.image_width = image_width
            self.image_height = image_height
            # Figure out image dimension
            self._build_IIR_filtering_graphs()
            ginit_op = tf.global_variables_initializer()
            linit_op = tf.local_variables_initializer()
            self.sess.run([ginit_op, linit_op])

            if self.load(checkpoint_dir):
                print("[*] Load Success")
            else:
                raise RuntimeError('MagNet: Failed to load checkpoint file.')
            self.is_graph_built = True
        try:
            i = int(self.ckpt_name.split('-')[-1])
            print("Iteration number is {:d}".format(i))
            vid_name = vid_name + '_' + str(i)
        except:
            print("Cannot get iteration number")

        if len(filter_a) is not 0:
            x_state = []
            y_state = []

            for frame in tqdm(vid_frames, desc='Applying IIR'):
                file_name = os.path.basename(frame)
                frame_no, _ = os.path.splitext(file_name)
                frame_no = int(frame_no)
                in_frames = [load_train_data([frame, frame, frame],
                             gray_scale=self.n_channels==1, is_testing=True)]
                in_frames = np.array(in_frames).astype(np.float32)

                texture_enc, x = self.sess.run([self.texture_enc, self.shape_rep],
                                               feed_dict={
                                                   self.input_image:
                                                   in_frames[:, :, :, :3],})
                x_state.insert(0, x)
                # set up initial condition.
                while len(x_state) < len(filter_b):
                    x_state.insert(0, x)
                if len(x_state) > len(filter_b):
                    x_state = x_state[:len(filter_b)]
                y = np.zeros_like(x)
                for i in range(len(x_state)):
                    y += x_state[i] * filter_b[i]
                for i in range(len(y_state)):
                    y -= y_state[i] * filter_a[i]
                # update y state
                y_state.insert(0, y)
                if len(y_state) > len(filter_a):
                    y_state = y_state[:len(filter_a)]

                out_amp = self.sess.run(self.output_image,
                                        feed_dict={self.out_texture_enc:
                                                     texture_enc,
                                                   self.filtered_enc: y,
                                                   self.ref_shape_enc: x,
                                                   self.amplification_factor:
                                                     [amplification_factor]})

                im_path = os.path.join(out_dir, file_name)
                out_amp = np.squeeze(out_amp)
                out_amp = (127.5*(out_amp+1)).astype('uint8')
                cv2.imwrite(im_path, cv2.cvtColor(out_amp,
                                                  code=cv2.COLOR_RGB2BGR))
        else:
            # This does FIR in fourier domain. Equivalent to cyclic
            # convolution.
            x_state = None
            for i, frame in tqdm(enumerate(vid_frames),
                                 desc='Getting encoding'):
                file_name = os.path.basename(frame)
                in_frames = [load_train_data([frame, frame, frame],
                                             gray_scale=self.n_channels==1, is_testing=True)]
                in_frames = np.array(in_frames).astype(np.float32)

                texture_enc, x = self.sess.run([self.texture_enc, self.shape_rep],
                                               feed_dict={
                                                   self.input_image:
                                                      in_frames[:, :, :, :3],})
                if x_state is None:
                    x_state = np.zeros(x.shape + (len(vid_frames),),
                                       dtype='float32')
                x_state[:, :, :, :, i] = x

            filter_fft = np.fft.fft(np.fft.ifftshift(filter_b),
                                    n=x_state.shape[-1])
            # Filtering
            for i in trange(x_state.shape[1], desc="Applying FIR filter"):
                x_fft = np.fft.fft(x_state[:, i, :, :], axis=-1)
                x_fft *= filter_fft[np.newaxis, np.newaxis, np.newaxis, :]
                x_state[:, i, :, :] = np.fft.ifft(x_fft)

            for i, frame in tqdm(enumerate(vid_frames), desc='Decoding'):
                file_name = os.path.basename(frame)
                frame_no, _ = os.path.splitext(file_name)
                frame_no = int(frame_no)
                in_frames = [load_train_data([frame, frame, frame],
                                             gray_scale=self.n_channels==1, is_testing=True)]
                in_frames = np.array(in_frames).astype(np.float32)
                texture_enc, _ = self.sess.run([self.texture_enc, self.shape_rep],
                                               feed_dict={
                                                   self.input_image:
                                                      in_frames[:, :, :, :3],
                                                         })
                out_amp = self.sess.run(self.output_image,
                                        feed_dict={self.out_texture_enc: texture_enc,
                                                   self.filtered_enc: x_state[:, :, :, :, i],
                                                   self.ref_shape_enc: x,
                                                   self.amplification_factor: [amplification_factor]})

                im_path = os.path.join(out_dir, file_name)
                out_amp = np.squeeze(out_amp)
                out_amp = (127.5*(out_amp+1)).astype('uint8')
                cv2.imwrite(im_path, cv2.cvtColor(out_amp,
                                                  code=cv2.COLOR_RGB2BGR))
            del x_state

        # Try to combine it into a video
        call([DEFAULT_VIDEO_CONVERTER, '-y', '-f', 'image2', '-r', '30', '-i',
              os.path.join(out_dir, '%06d.png'), '-c:v', 'libx264',
              os.path.join(out_dir, vid_name + '.mp4')]
            )

    # Training code.
    def _build_training_graph(self, train_config):
        self.global_step = tf.Variable(0, trainable=False)
        filename_queue = tf.train.string_input_producer(
                            [os.path.join(train_config["dataset_dir"],
                                          'train.tfrecords')],
                            num_epochs=train_config["num_epochs"])
        frameA, frameB, frameC, frameAmp, amplification_factor = \
            read_and_decode_3frames(filename_queue,
                                    (train_config["image_height"],
                                     train_config["image_width"],
                                     self.n_channels))
        min_after_dequeue = 1000
        num_threads = 16
        capacity = min_after_dequeue + \
            (num_threads + 2) * train_config["batch_size"]

        frameA, frameB, frameC, frameAmp, amplification_factor = \
            tf.train.shuffle_batch([frameA,
                                    frameB,
                                    frameC,
                                    frameAmp,
                                    amplification_factor],
                                   batch_size=train_config["batch_size"],
                                   capacity=capacity,
                                   num_threads=num_threads,
                                   min_after_dequeue=min_after_dequeue)

        frameA = preprocess_image(frameA, train_config)
        frameB = preprocess_image(frameB, train_config)
        frameC = preprocess_image(frameC, train_config)
        self.loss_function = partial(self._loss_function,
                                     train_config=train_config)
        self.output = self.image_transformer(frameA,
                                             frameB,
                                             amplification_factor,
                                             [train_config["image_height"],
                                              train_config["image_width"]],
                                             self.arch_config, True, False)
        self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if self.reg_loss and train_config["weight_decay"] > 0.0:
            print("Adding Regularization Weights.")
            self.loss = self.loss_function(self.output, frameAmp) + \
                train_config["weight_decay"] * tf.add_n(self.reg_loss)
        else:
            print("No Regularization Weights.")
            self.loss = self.loss_function(self.output, frameAmp)
        # Add regularization more
        # TODO: Hardcoding the network name scope here.
        with tf.variable_scope('ynet_3frames/encoder', reuse=True):
            texture_c, shape_c = self._encoder(frameC)
            self.loss = self.loss + \
                train_config["texture_loss_weight"] * L1_loss(texture_c, self.texture_a) + \
                train_config["shape_loss_weight"] * L1_loss(shape_c, self.shape_b)

        self.loss_sum = tf.summary.scalar('train_loss', self.loss)
        self.image_sum = tf.summary.image('train_B_OUT',
                                          tf.concat([frameB, self.output],
                                                    axis=2),
                                          max_outputs=2)
        if self.n_channels == 3:
            self.image_comp_sum = tf.summary.image('train_GT_OUT',
                                                   frameAmp - self.output,
                                                   max_outputs=2)
            self.image_orig_comp_sum = tf.summary.image('train_ORIG_OUT',
                                                        frameA - self.output,
                                                        max_outputs=2)
        else:
            self.image_comp_sum = tf.summary.image('train_GT_OUT',
                                                   tf.concat([frameAmp,
                                                              self.output,
                                                              frameAmp],
                                                             axis=3),
                                                   max_outputs=2)
            self.image_orig_comp_sum = tf.summary.image('train_ORIG_OUT',
                                                        tf.concat([frameA,
                                                                   self.output,
                                                                   frameA],
                                                                  axis=3),
                                                        max_outputs=2)
        self.saver = tf.train.Saver(max_to_keep=train_config["ckpt_to_keep"])

    # Loss function
    def _loss_function(self, a, b, train_config):
        # Use train_config to implement more advance losses.
        with tf.variable_scope("loss_function"):
            return L1_loss(a, b) * train_config["l1_loss_weight"]

    def train(self, train_config):
        # Define training graphs
        self._build_training_graph(train_config)

        self.lr = tf.train.exponential_decay(train_config["learning_rate"],
                                             self.global_step,
                                             train_config["decay_steps"],
                                             train_config["lr_decay"],
                                             staircase=True)
        self.optim_op = tf.train.AdamOptimizer(self.lr,
                                               beta1=train_config["beta1"]) \
            .minimize(self.loss,
                      var_list=tf.trainable_variables(),
                      global_step=self.global_step)

        ginit_op = tf.global_variables_initializer()
        linit_op = tf.local_variables_initializer()
        self.sess.run([ginit_op, linit_op])

        self.writer = tf.summary.FileWriter(train_config["logs_dir"],
                                            self.sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        start_time = time.time()
        for v in tf.trainable_variables():
            print(v)
        if train_config["continue_train"] and \
                self.load(train_config["checkpoint_dir"]):
            print('[*] Load Success')
        elif train_config["restore_dir"] and \
                self.load(train_config["restore_dir"],
                          tf.train.Saver(var_list=tf.trainable_variables())):
            self.sess.run(self.global_step.assign(0))
            print('[*] Restore success')
        else:
            print('Training from scratch.')
        try:
            while not coord.should_stop():
                _, loss_sum_str = self.sess.run([self.optim_op, self.loss_sum])
                global_step = self.sess.run(self.global_step)
                self.writer.add_summary(loss_sum_str, global_step)

                if global_step % 100 == 0:
                    # Write image summary.
                    img_sum_str, img_comp_str, img_orig_str = \
                            self.sess.run([self.image_sum,
                                           self.image_comp_sum,
                                           self.image_orig_comp_sum])
                    self.writer.add_summary(img_sum_str, global_step)
                    self.writer.add_summary(img_comp_str, global_step)
                    self.writer.add_summary(img_orig_str, global_step)

                elapsed_time = time.time() - start_time
                print ("Steps: %2d time: %4.4f (%4.4f steps/sec)" % (
                    global_step, elapsed_time,
                    float(global_step) / elapsed_time))

                if np.mod(global_step, train_config["save_freq"]) == 2:
                    self.save(train_config["checkpoint_dir"], global_step)

        except tf.errors.OutOfRangeError:
            print('Done Training.')
        finally:
            coord.request_stop()
            coord.join(threads)

