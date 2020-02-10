# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SpeechCommands dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import numpy as np

from pydub import AudioSegment
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@article{speechcommandsv2,
   author = {{Warden}, P.},
    title = "{Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.03209},
  primaryClass = "cs.CL",
  keywords = {Computer Science - Computation and Language, Computer Science - Human-Computer Interaction},
    year = 2018,
    month = apr,
    url = {https://arxiv.org/abs/1804.03209},
}
"""

_DESCRIPTION = """
An audio dataset of spoken words designed to help train and evaluate keyword 
spotting systems. Its primary goal is to provide a way to build and test small
models that detect when a single word is spoken, from a set of ten target words, 
with as few false positives as possible from background noise or unrelated 
speech.
"""

_DOWNLOAD_PATH = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
_TEST_DOWNLOAD_PATH_ = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'

_SPLITS = ['train', 'valid', 'test']

WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
SILENCE = '_silence_'
UNKNOWN = '_unknown_'
BACKGROUND_NOISE = '_background_noise_'
SAMPLE_RATE = 16000


class SpeechCommands(tfds.core.GeneratorBasedBuilder):
  """The Speech Commands dataset for keyword detection."""

  VERSION = tfds.core.Version('0.0.2')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            'audio': tfds.features.Audio(),
            'label': tfds.features.ClassLabel(names=WORDS + [SILENCE, UNKNOWN])
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=('audio', 'label'),
        # Homepage of the dataset for documentation
        homepage='https://arxiv.org/abs/1804.03209',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    dl_path, dl_test_path = dl_manager.download_and_extract(
        [_DOWNLOAD_PATH, _TEST_DOWNLOAD_PATH_])

    # The main tar file already contains all of the test files, except fot the
    # silence ones. In fact it does not contain silence files at all. So for the
    # test set we take the silence files from the test tar file, while for train
    # and validation we build them from the _background_noise_ folder.
    test_split = os.path.join(dl_path, 'testing_list.txt')
    validation_split = os.path.join(dl_path, 'validation_list.txt')

    # We don't use these files to generate the examples, just to remove it from
    # the train set.
    with tf.io.gfile.GFile(test_split) as f:
      train_test_paths = f.read().strip().splitlines()
    train_test_paths = [os.path.join(dl_path, p) for p in train_test_paths]
    with tf.io.gfile.GFile(validation_split) as f:
      validation_paths = f.read().strip().splitlines()
    validation_paths = [os.path.join(dl_path, p) for p in validation_paths]

    # Original validation files did include silence - we add them manually here
    validation_paths.append(
        os.path.join(dl_path, BACKGROUND_NOISE, 'running_tap.wav'))

    train_paths = glob.glob(os.path.join(dl_path, '*', '*.wav'))
    train_paths = set(train_paths)

    # The paths for the train set is just whichever paths that do not exist in
    # either the test or validation splits.
    for p in validation_paths:
      train_paths.remove(p)
    for p in train_test_paths:
      train_paths.remove(p)

    # These are the actual paths for the test set.
    test_paths = glob.glob(os.path.join(dl_test_path, '*', '*.wav'))
    test_paths = set(test_paths)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'paths': train_paths},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'paths': validation_paths},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'paths': test_paths},
        ),
    ]

  def _generate_examples(self, paths):
    """Yields examples."""
    for path in paths:
      relpath, wavname = os.path.split(path)
      _, word = os.path.split(relpath)
      example_id = '{}_{}'.format(word, wavname)
      if word in WORDS:
        label = word
      elif word == SILENCE or word == BACKGROUND_NOISE:
        label = SILENCE
      else:
        # Note that in the train and validation there are a lot more _unknown_
        # labels than any of the other ones.
        label = UNKNOWN

      if word == BACKGROUND_NOISE:
        # Special handling of background noise. We need to cut these files to
        # many small files with 1 seconds length, and transform it to silence.
        audio_samples = np.array(
            AudioSegment.from_file(path, format='wav').get_array_of_samples())

        for start in range(0,
                           len(audio_samples) - SAMPLE_RATE, SAMPLE_RATE // 2):
          audio_segment = audio_samples[start:start + SAMPLE_RATE]
          cur_id = '{}_{}'.format(example_id, start)
          example = {'audio': audio_segment, 'label': label}
          yield cur_id, example
      else:
        example = {
            'audio': path,
            'label': label,
        }
        yield example_id, example
