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

"""Unit tests for screen_recorder.py."""

import logging
import os
import platform
import random
import socket
import sys
import time
import unittest
from unittest import mock

import cv2
import numpy as np
from absl import flags
from absl.testing import parameterized
from mobly import logger, runtime_test_info
from mobly.controllers.android_device_lib import service_manager
from mobly.controllers.android_device_lib.services import screen_recorder

_MOCK_SERIAL = 'DEADBEEF'
_WIDTH = 768
_HEIGHT = 1080
_AUDIO_SIZE = 1024

# Create a blank PNG stream in memory to use as a mock frame.
_BLACK_IMAGE_ARRAY = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
_BLANK_PNG_STREAM = cv2.imencode('.png', _BLACK_IMAGE_ARRAY)[1].tobytes()
_ACTUAL_FRAME_SIZE = len(_BLANK_PNG_STREAM)


def mock_adb_shell(args, timeout=None):
    """Mocks the adb shell command."""
    del timeout  # unused
    if args == 'echo $(date +%Y-%m-%dT%T)${EPOCHREALTIME:10:4}':
        return b'2020-01-01T00:00:00.000'
    if args == 'date +"%s.%N"':
        return b'2021-01-01T00:00:00.000'
    return bytes()


def mock_adb_forward(args, timeout=None):
    """Mocks the adb forward command."""
    del args, timeout  # unused
    return b'12345\n'


def mock_device():
    """Creates a mock AndroidDevice object."""
    device = mock.Mock()
    device.log = logging.Logger('test')
    device.serial = _MOCK_SERIAL
    device.debug_tag = _MOCK_SERIAL

    def mock_generate_filename(prefix, extension_name='mp4'):
        return f'{prefix},DEADBEEF,0.{extension_name}'

    device.generate_filename.side_effect = mock_generate_filename
    device.adb = mock.Mock()
    device.adb.shell = mock.Mock(side_effect=mock_adb_shell)
    device.adb.pull = mock.Mock()
    device.adb.push = mock.Mock()
    device.adb.forward = mock.Mock(side_effect=mock_adb_forward)

    device.services = service_manager.ServiceManager(device)
    return device


def mock_server_output(args, **kwargs):
    """Mocks the server output."""
    del args, kwargs  # unused
    subpro = mock.Mock()
    subpro.stdout = mock.Mock()
    subpro.kill = mock.Mock()
    subpro.stdout.readline.side_effect = [
        b'[server] INFO: Device: DEADBEEF.\n',
        b'xxx',
    ]
    return subpro


def mock_recv(k_bytes, portrait=True):
    """Mocks the socket recv method to simulate receiving data."""
    if k_bytes == 1:  # dummy byte for video socket
        return b'n'
    if k_bytes == 4:  # dummy byte for audio socket
        return b' raw'
    if k_bytes == screen_recorder._DEVICE_META_DATA_SIZE:
        ret_bytes = 'Mi 10'.encode('utf-8')
        ret_bytes += b'\x00' * (
            screen_recorder._DEVICE_META_DATA_SIZE - len(ret_bytes)
        )
        return ret_bytes
    if k_bytes == screen_recorder._VIDEO_META_DATA_SIZE:
        ret_bytes = 'Mi 10'.encode('utf-8')
        ret_bytes += b'\x00' * (
            screen_recorder._VIDEO_META_DATA_SIZE - 4 - len(ret_bytes)
        )
        ret_bytes += int.to_bytes(768 if portrait else 1080, 2, 'big')
        ret_bytes += int.to_bytes(1080 if portrait else 768, 2, 'big')
        return ret_bytes
    if k_bytes == screen_recorder._FRAME_HEADER_SIZE:  # a fake frame head
        sleep_time = random.randint(100, 200)  # monitor frame delay
        time.sleep(sleep_time / 1000)
        curtime = time.monotonic()
        ret_bytes = bytes()
        ret_bytes += int.to_bytes(
            _WIDTH if portrait else _HEIGHT, 4, 'big'
        )  # width
        ret_bytes += int.to_bytes(
            _HEIGHT if portrait else _WIDTH, 4, 'big'
        )  # height
        ret_bytes += int.to_bytes(1, 4, 'big')
        ret_bytes += int.to_bytes(int(curtime * 1e9), 8, 'big')  # timestamp
        # Use the actual size of the PNG file in the header.
        ret_bytes += int.to_bytes(_ACTUAL_FRAME_SIZE, 4, 'big')  # bytes
        return ret_bytes
    if k_bytes == screen_recorder._AUDIO_HEADER_SIZE:  # a fake audio header
        ret_bytes = bytes()
        ret_bytes += int.to_bytes(
            int(time.monotonic() * 1e9), 8, 'big'
        )  # timestamp
        ret_bytes += int.to_bytes(_AUDIO_SIZE, 4, 'big')  # bytes
        return ret_bytes
    if k_bytes == _ACTUAL_FRAME_SIZE:
        return _BLANK_PNG_STREAM
    return b'\x00' * k_bytes


def mock_recv_landscape_frame(k_bytes):
    """Mocks receiving a landscape frame."""
    return mock_recv(k_bytes, portrait=False)


def mock_recv_large_frame(k_bytes):
    """Mocks getting large frame size meta and raising service exception."""
    if k_bytes == screen_recorder._FRAME_HEADER_SIZE:
        sleep_time = random.randint(100, 200)  # monitor frame delay
        time.sleep(sleep_time / 1000)
        curtime = time.monotonic()
        ret_bytes = bytes()
        ret_bytes += int.to_bytes(768, 4, 'big')  # width
        ret_bytes += int.to_bytes(1080, 4, 'big')  # height
        ret_bytes += int.to_bytes(1, 4, 'big')
        ret_bytes += int.to_bytes(int(curtime * 1e9), 8, 'big')  # timestamp
        # large bytes causing service exception
        ret_bytes += int.to_bytes(int(2e9), 4, 'big')
        return ret_bytes
    return mock_recv(k_bytes)


def mock_socket_output(arg1, arg2, **kwargs):
    """Creates a mock socket that correctly handles being closed."""
    del arg1, arg2, kwargs  # unused
    sock = mock.Mock()
    sock._is_closed = False

    def mock_close():
        """Sets the internal closed flag to true."""
        sock._is_closed = True

    sock.close.side_effect = mock_close

    def mock_recv_wrapper(k_bytes):
        """Raises socket.timeout if closed, otherwise calls the real mock."""
        if sock._is_closed:
            raise socket.timeout('Mock socket closed')
        return mock_recv(k_bytes)

    sock.recv.side_effect = mock_recv_wrapper

    sock.setsockopt = mock.Mock()
    return sock


class RecordingServiceTest(parameterized.TestCase):
    """Unit tests for the ScreenRecorder service."""

    def setUp(self):
        """Sets up the test environment."""
        super().setUp()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(
            logger,
            'get_log_file_timestamp',
            autospec=True,
            side_effect=map(str, range(20)),
        ).start()
        mock.patch.object(
            screen_recorder.subprocess, 'Popen', side_effect=mock_server_output
        ).start()
        mock.patch.object(
            screen_recorder.socket, 'socket', side_effect=mock_socket_output
        ).start()

        mock.patch.object(platform, 'system', return_value='Linux').start()
        mock.patch.object(
            screen_recorder.shutil, 'which', return_value='/usr/bin/ffmpeg'
        ).start()
        mock.patch.object(cv2, 'VideoWriter', mock.Mock()).start()
        wave_mock = mock.Mock()
        wave_mock.setnchannels = mock.Mock()
        wave_mock.setsampwidth = mock.Mock()
        wave_mock.setframerate = mock.Mock()
        wave_mock.writeframes = mock.Mock()
        wave_mock.close = mock.Mock()
        mock.patch.object(
            screen_recorder.wave, 'open', return_value=wave_mock
        ).start()
        mock.patch.object(
            screen_recorder.subprocess, 'run', mock.Mock()
        ).start()
        mock.patch.object(screen_recorder.shutil, 'move', mock.Mock()).start()

    def set_device(self, **kwargs):
        """Sets up a mock device with the screen recorder service."""
        device = mock_device()
        device.log_path = 'fake_log_path'
        args_dic = {
            'output_dir': '/tmp',
            'video_bit_rate': 1000000,
            'max_fps': 80,
            'max_video_size': 1080,
            'raise_when_no_frames': False,
            'display_id': None,
            'enable_audio': False,
        }
        # Set config args in this function instead of each test
        for key, value in args_dic.items():
            if key not in kwargs:
                kwargs[key] = value
        device.services.register(
            'screen_recorder',
            screen_recorder.ScreenRecorder,
            screen_recorder.Configs(**kwargs),
            start_service=False,
        )
        return device

    def test_register_without_config(self):
        """Tests that the service can be registered without a config."""
        device = mock_device()
        device.log_path = 'fake_log_path'

        device.services.register(
            'screen_recorder', screen_recorder.ScreenRecorder, start_service=False
        )

        self.assertEqual(
            repr(device.services.screen_recorder),
            (
                'ScreenRecorder(serial=DEADBEEF '
                'dir=fake_log_path bit_rate=100000 max_height=1080)'
            ),
        )

    def test_start_server_success(self):
        """Tests that the server starts successfully."""
        device = self.set_device()
        self.addCleanup(device.services.screen_recorder.stop)

        device.services.screen_recorder._start_server()

        self.assertEqual(
            repr(device.services.screen_recorder),
            (
                'ScreenRecorder(serial=DEADBEEF '
                'dir=/tmp bit_rate=1000000 max_height=1080)'
            ),
        )

    def test_service_start(self):
        """Tests that the service starts successfully."""
        output_dir = self.create_tempdir()
        device = self.set_device(
            output_dir=output_dir.full_path, raise_when_no_frames=False
        )
        self.addCleanup(device.services.screen_recorder.stop)

        device.services.screen_recorder.start()

        self.assertTrue(device.services.screen_recorder.is_alive)
        self.assertTrue(device.services.screen_recorder._first_run)
        self.assertTrue(device.services.screen_recorder._timestamps.use_frame_time)
        device.services.screen_recorder.stop()

    @mock.patch.object(screen_recorder.subprocess, 'Popen')
    def test_start_server_fail(self, mock_popen):
        """Tests that the service raises an error if the server fails to start."""
        device = self.set_device()
        self.addCleanup(device.services.screen_recorder.stop)
        mock_popen.return_value.stdout.readline.side_effect = [
            b'[server] ERROR: Server failed to start.\n',
            b'',  # Ensure the readline loop terminates
        ]

        with self.assertRaisesRegex(
            screen_recorder.Error, 'Start server failed.'
        ):
            device.services.screen_recorder.start()

    @mock.patch('os.remove')
    def test_multiple_records(self, mock_os_remove):
        """Tests that multiple records only generate one file."""
        output_dir = self.create_tempdir().full_path
        device = self.set_device(
            output_dir=output_dir,
            raise_when_no_frames=False,
            enable_audio=True,
            save_audio_file=True,
        )
        self.addCleanup(device.services.screen_recorder.stop)
        mock_record = mock.Mock()
        mock_record.begin_time = 123
        mock_record.signature = 'test_foo-123'
        run_dir = self.create_tempdir()
        test_run_info = runtime_test_info.RuntimeTestInfo(
            'test_foo', run_dir.full_path, mock_record
        )

        # call `start()` and `stop()` twice to test multiple records only generate
        # one file.
        with self.assertLogs(device.log, level='INFO') as log_output_1:
            device.services.screen_recorder.start()
        time.sleep(0.5)  # Let the service record a few frames
        device.services.screen_recorder.stop()
        time.sleep(0.1)

        # The temporary files should be created.
        temp_video_file = os.path.join(
            output_dir, 'only_video,DEADBEEF,0.mp4'
        )
        temp_audio_file = os.path.join(
            output_dir, 'only_audio,DEADBEEF,0.wav'
        )
        with open(temp_video_file, 'w', encoding='utf-8') as f:
            f.write('')
        with open(temp_audio_file, 'w', encoding='utf-8') as f:
            f.write('')

        with self.assertLogs(device.log, level='DEBUG') as log_output_2:
            device.services.screen_recorder.start()
        time.sleep(0.1)
        device.services.screen_recorder.stop()

        file_path_list = (
            device.services.screen_recorder.create_output_excerpts(
                test_run_info
            )
        )

        # Multiple records should be added frames into one file in
        # `create_output_excerpts()`.
        # Video file list should be cleared
        self.assertLen(file_path_list, 2)  # Expecting 2 files: video and audio
        self.assertEmpty(device.services.screen_recorder._video_files)
        # Ensure video start info is only logged once.
        self.assertIn(
            'INFO:DEADBEEF Start video recording 2020-01-01T00:00:00.000, output'
            ' filename video,DEADBEEF,0.mp4',
            ' '.join(log_output_1.output),
            'Video starting message is absent in log output.',
        )
        self.assertNotIn(
            'INFO:DEADBEEF Start video recording 2020-01-01T00:00:00.000, output'
            ' filename video,DEADBEEF,0.mp4',
            ' '.join(log_output_2.output),
            'Video starting message is present in log output.',
        )
        device.services.screen_recorder.stop()

        mock_os_remove.assert_any_call(temp_video_file)

    @mock.patch('os.remove')
    def test_video_recording_and_excerpts(self, mock_os_remove):
        """Tests the entire video recording and excerpt generation process."""
        output_dir = self.create_tempdir().full_path
        device = self.set_device(
            output_dir=output_dir,
            raise_when_no_frames=False,
            enable_audio=True,
            save_audio_file=True,
        )
        self.addCleanup(device.services.screen_recorder.stop)
        mock_record = mock.Mock()
        mock_record.begin_time = 123
        mock_record.signature = 'test_foo-123'
        run_dir = self.create_tempdir()
        test_run_info = runtime_test_info.RuntimeTestInfo(
            'test_foo', run_dir.full_path, mock_record
        )

        device.services.screen_recorder.start()
        time.sleep(0.5)  # Let the service record a few frames

        # The temporary files should be created.
        temp_video_file = os.path.join(
            output_dir, 'only_video,DEADBEEF,0.mp4'
        )
        temp_audio_file = os.path.join(
            output_dir, 'only_audio,DEADBEEF,0.wav'
        )
        with open(temp_video_file, 'w', encoding='utf-8') as f:
            f.write('')
        with open(temp_audio_file, 'w', encoding='utf-8') as f:
            f.write('')

        device.services.screen_recorder.stop()

        file_path_list = (
            device.services.screen_recorder.create_output_excerpts(
                test_run_info
            )
        )

        self.assertLen(file_path_list, 2)
        self.assertEqual(
            file_path_list[0],
            os.path.join(test_run_info.output_path, 'video,DEADBEEF,0.mp4'),
        )
        self.assertEqual(
            file_path_list[1],
            os.path.join(
                test_run_info.output_path, 'only_audio,DEADBEEF,0.wav'
            ),
        )
        mock_os_remove.assert_any_call(temp_video_file)

    @mock.patch.object(screen_recorder.subprocess, 'Popen')
    def test_start_server_with_audio_command(self, mock_popen):
        """Tests that the server starts with audio-related commands."""
        device = self.set_device(enable_audio=True)
        self.addCleanup(device.services.screen_recorder.stop)
        mock_popen.side_effect = mock_server_output

        device.services.screen_recorder._start_server()

        mock_popen.assert_called_once()
        args, _ = mock_popen.call_args
        cmd_list = args[0]
        self.assertIn('audio=true', cmd_list)
        self.assertIn('audio_codec=raw', cmd_list)
        self.assertIn('audio_source=mic', cmd_list)
        self.assertNotIn('audio=false', cmd_list)

    @mock.patch(
        'mobly.controllers.android_device_lib.services.screen_recorder.shutil.which',
        return_value=None,
    )
    def test_audio_disabled_if_ffmpeg_missing(self, mock_shutil_which):
        """Tests audio is disabled if ffmpeg lib is not available."""
        del mock_shutil_which  # unused
        # Register the service after patching ffmpeg_lib to None.
        device = mock_device()
        output_dir = self.create_tempdir().full_path
        with self.assertRaisesRegex(
            screen_recorder.Error, 'ffmpeg not found in system PATH.'
        ):
            device.services.register(
                'screen_recorder',
                screen_recorder.ScreenRecorder,
                screen_recorder.Configs(
                    output_dir=output_dir, enable_audio=True
                ),
                start_service=False,
            )

    @mock.patch.object(screen_recorder.socket, 'socket')
    def test_start_audio_connection_fail(self, mock_socket_constructor):
        """Tests that a ConnectionError is raised on audio socket failure."""
        mock_socket_instance = mock.Mock()
        mock_socket_instance.recv.return_value = b''
        mock_socket_constructor.return_value = mock_socket_instance

        device = self.set_device(enable_audio=True)
        self.addCleanup(device.services.screen_recorder.stop)

        with self.assertRaisesRegex(
            ConnectionError, 'Failed to build audio socket connection'
        ):
            device.services.screen_recorder._start_audio_connection()

    @mock.patch.object(screen_recorder.subprocess, 'run')
    def test_combine_files_fail_with_ffmpeg_error(self, mock_subprocess_run):
        """Tests error handling when ffmpeg fails to combine files."""
        output_dir = self.create_tempdir().full_path
        device = self.set_device(
            output_dir=output_dir,
            raise_when_no_frames=False,
            enable_audio=True,
            save_audio_file=True,
        )
        self.addCleanup(device.services.screen_recorder.stop)
        mock_subprocess_run.side_effect = (
            screen_recorder.subprocess.CalledProcessError(
                returncode=1, cmd='ffmpeg', stderr=b'ffmpeg error'
            )
        )

        device.services.screen_recorder.start()
        time.sleep(0.2)
        device.services.screen_recorder.stop()

        with self.assertRaisesRegex(
            screen_recorder.Error,
            'FFmpeg merging failed with error: ffmpeg error',
        ):
            device.services.screen_recorder._release_final_output_video_file()

    @mock.patch.object(
        screen_recorder.ScreenRecorder, '_handle_socket_disconnection'
    )
    @mock.patch.object(screen_recorder.ScreenRecorder, '_recv_bytes')
    def test_socket_disconnection_in_audio_loop(
        self,
        mock_recv_bytes,
        mock_handle_socket_disconnection,
    ):
        """Tests that a socket disconnection in the audio loop triggers a restart."""
        device = self.set_device(enable_audio=True)
        self.addCleanup(device.services.screen_recorder.stop)

        # Simulate the audio socket returning no data, which indicates a
        # disconnection.
        def recv_side_effect(stream_socket, k_bytes):
            if (
                stream_socket
                == device.services.screen_recorder._audio_socket
            ):
                return b''  # Empty bytes simulates a closed socket
            return mock_recv(k_bytes)  # Use original mock for video socket

        mock_recv_bytes.side_effect = recv_side_effect

        device.services.screen_recorder.start()
        time.sleep(0.2)  # Allow time for the loop to run and fail

        # The disconnection handler should have been called.
        mock_handle_socket_disconnection.assert_called()

        device.services.screen_recorder.stop()

    @parameterized.named_parameters(
        ('mic_source', screen_recorder.AudioSourceType.MIC),
        (
            'mic_voice_communication_source',
            screen_recorder.AudioSourceType.MIC_VOICE_COMMUNICATION,
        ),
        (
            'playback_source',
            screen_recorder.AudioSourceType.PLAYBACK,
        ),
        (
            'voice_call_down_link_source',
            screen_recorder.AudioSourceType.VOICE_CALL_DOWN_LINK,
        ),
        ('output_source', screen_recorder.AudioSourceType.OUTPUT),
    )
    @mock.patch.object(screen_recorder.subprocess, 'Popen')
    def test_start_server_with_audio_source(self, source, mock_popen):
        """Tests the server starts with the correct audio source command."""
        device = self.set_device(enable_audio=True, audio_source=source)
        self.addCleanup(device.services.screen_recorder.stop)
        mock_popen.side_effect = mock_server_output

        device.services.screen_recorder._start_server()

        mock_popen.assert_called_once()
        args, _ = mock_popen.call_args
        cmd_list = args[0]

        # Check that general audio commands are present
        self.assertIn('audio=true', cmd_list)
        self.assertIn('audio_codec=raw', cmd_list)

        # Check that the correct, specific audio source is in the command
        self.assertIn(f'audio_source={source.value}', cmd_list)

    @mock.patch.object(screen_recorder.subprocess, 'Popen')
    def test_command_defaults_to_mic_source_when_unspecified(
        self,
        mock_popen,
    ):
        """Tests the default 'audio_source=mic' when not specified in the config."""
        mock_popen.side_effect = mock_server_output

        # Create the service with audio enabled, but not specify audio_source.
        device = self.set_device(enable_audio=True)
        self.addCleanup(device.services.screen_recorder.stop)

        device.services.screen_recorder._start_server()

        mock_popen.assert_called_once()
        args, _ = mock_popen.call_args
        final_command_list = args[0]

        # Assert that the default source was correctly added to the final command.
        self.assertIn('audio_source=mic', final_command_list)

    @mock.patch('os.remove')
    def test_audio_file_saved_when_save_audio_file_is_true(
        self,
        mock_os_remove,
    ):
        """Tests that the temp audio file is saved and temp video file is removed after generating excerpts."""
        output_dir = self.create_tempdir().full_path
        device = self.set_device(
            output_dir=output_dir,
            raise_when_no_frames=False,
            enable_audio=True,
            save_audio_file=True,
        )
        self.addCleanup(device.services.screen_recorder.stop)
        mock_record = mock.Mock()
        mock_record.begin_time = 123
        mock_record.signature = 'test_foo-123'
        run_dir = self.create_tempdir()
        test_run_info = runtime_test_info.RuntimeTestInfo(
            'test_foo', run_dir.full_path, mock_record
        )

        device.services.screen_recorder.start()

        time.sleep(0.5)  # Let the service record a few frames

        # The temporary files should be created.
        temp_video_file = os.path.join(
            output_dir, 'only_video,DEADBEEF,0.mp4'
        )
        temp_audio_file = os.path.join(
            output_dir, 'only_audio,DEADBEEF,0.wav'
        )
        with open(temp_video_file, 'w', encoding='utf-8') as f:
            f.write('')
        with open(temp_audio_file, 'w', encoding='utf-8') as f:
            f.write('')

        file_path_list = (
            device.services.screen_recorder.create_output_excerpts(
                test_run_info
            )
        )

        self.assertLen(file_path_list, 2)
        self.assertEqual(
            file_path_list[0],
            os.path.join(test_run_info.output_path, 'video,DEADBEEF,0.mp4'),
        )
        self.assertEqual(
            file_path_list[1],
            os.path.join(
                test_run_info.output_path, 'only_audio,DEADBEEF,0.wav'
            ),
        )
        device.services.screen_recorder.stop()

        mock_os_remove.assert_any_call(temp_video_file)


if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    unittest.main()
