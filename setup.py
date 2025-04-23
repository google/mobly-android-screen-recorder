# Copyright 2024 Google LLC
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

"""Installs Mobly Android Screen Recorder."""

import setuptools

install_requires = [
    'mobly>=1.12.2',
    'numpy>=2.2.1',
    'opencv-python>=4.10.0.84',
    'retrying>=1.3.4',
]

setuptools.setup(
    name='mobly-android-screen-recorder',
    version='1.0.1',
    author='Yao Yao',
    author_email='yayao@google.com',
    description='Mobly Android Screen Recorder service for using Python code to screencast the Android devices in Mobly tests.',
    license='Apache2.0',
    url='https://github.com/google/mobly-android-screen-recorder',
    packages=setuptools.find_namespace_packages(include=['mobly.controllers.android_device_lib.services*']),
    package_data={'mobly.controllers.android_device_lib.services': ['data/*']},
    install_requires=install_requires,
    python_requires='>=3.11',
)
