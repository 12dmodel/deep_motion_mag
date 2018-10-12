# Learning-based Video Motion Magnification

Tensorflow implementation of Learning-based Video Motion Magnification. [\[Project page\]](https://people.csail.mit.edu/tiam/deepmag/) [\[Paper\]](https://arxiv.org/abs/1804.02684) [\[Data\]](https://groups.csail.mit.edu/graphics/deep_motion_mag/data/readme.txt)

Collaborators: 
\*(Tae-Hyun Oh, Ronnachai "Tiam" Jaroensri), Changil Kim, Mohamed A. Elgharib, Fr&eacute;do Durand, William T. Freeman, Wojciech Matusik

\*Equal contribution.

[Video demo]

[![Video demo](https://img.youtube.com/vi/GrMLeEcSNzY/0.jpg)](https://www.youtube.com/watch?v=GrMLeEcSNzY)

# Installation

Install required packages:

```
sudo apt-get install python-dev 	# required to install setproctitle
pip install -r requirements.txt
```

This code has been tested with Tensorflow 1.3 and 1.8, CUDA 8.5 and 9.1, Ubuntu 14.04 and 16.04, although we expect it to work with any newer versions of Tensorflow and CUDA.


# Running

Download and place the [pre-trained model](https://people.csail.mit.edu/tiam/deepmag/data.zip) in to `data/`.
Only checkpoint is included in the package, and should be under `data/training/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3/checkpoint`.

Place video frames under `data/vids/<video_name>`, and run one of the two scripts:

```
# Two-frame based processing (Static, Dynamic mode)
sh run_on_test_videos.sh <experiment_name> <video_name> <amplification_factor> <run_dynamic_mode>
# Temporal filtering based processing (frequency band pass filter)
sh run_temporal_on_test_videos.sh <experiment_name> <video_name> <amplification_factor> <low_cutoff> <high_cutoff> <sampling_rate> <n_filter_tap> <filter_type>
```

Here, Dynamic mode stands for consequtive frame processing.
Our code supports {"fir", "butter", "differenceOfIIR"} filters. For FIR filters, the filtering could be done both in Fourier domain and in temporal domain.

For example, for the [baby video](https://people.csail.mit.edu/mrub/evm/video/baby.mp4), you need to extract video frames into PNG files.
You can run these commands from the repo directory (assuming the frames are under `data/vids/baby`):

```
# Static mode, using first frame as reference.
sh run_on_test_videos.sh o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3 baby 10
# Dynamic mode, magnify difference between consecutive frames.
sh run_on_test_videos.sh o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3 baby 10 yes
# Using temporal filter (same as Wadhawa et al.)
sh run_temporal_on_test_videos.sh o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3 baby 20 0.04 0.4 30 2 differenceOfIIR
```

For your reference, you can use this command to extract `baby` video frames (replace `avconv` with `ffmpeg` as necessary):
```
avconv -i <path_to_input>/baby.mp4 -f image2 <path_to_output>/baby/%06d.png
```

For custom video and output location, you should look into modifying the two scripts directly. For custom checkpoint location, look in the configuration file under `configs/`. There should be an option for checkpoint location that you can edit.


# Training

Use the `convert_3frames_data_to_tfrecords.py` utilities to convert your dataset into `tfrecords`, or download the pre-processed `tfrecords` (see [here](https://groups.csail.mit.edu/graphics/deep_motion_mag/data/readme.txt)).

Set the data path in the config file, and run the training script:

```
bash train.sh configs/<your config file>.conf
```

Be sure to configure your GPU appropriately (e.g. `export CUDA_VISIBLE_DEVICES=...`).

# Citation

If you use this code, please cite our paper:

```
@article{oh2018learning,
  title={Learning-based Video Motion Magnification},
  author={Oh, Tae-Hyun and Jaroensri, Ronnachai and Kim, Changil and Elgharib, Mohamed and Durand, Fr{\'e}do and Freeman, William T and Matusik, Wojciech},
  journal={arXiv preprint arXiv:1804.02684},
  year={2018}
}
```

This implementation is provided for academic use only. 

# Acknowledgement

The authors would like to thank Toyota Research Institute, Shell Research, and Qatar Computing Research Institute for their generous support of this project. Changil Kim was supported by a Swiss National Science Foundation fellowship P2EZP2 168785.

Utilities and skeleton code borrowed from [tensorflow implementation of CycleGAN](https://github.com/xhujoy/CycleGAN-tensorflow)
