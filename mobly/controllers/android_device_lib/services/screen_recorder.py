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

"""An AndroidDevice service for recording screen."""

import dataclasses
import enum
import os
import shutil
import socket
import subprocess
import threading
import time
import wave
from concurrent import futures
from typing import Any, List, Optional

import cv2
import numpy as np
import retrying
from mobly import runtime_test_info
from mobly.controllers import android_device
from mobly.controllers.android_device_lib import adb, errors
from mobly.controllers.android_device_lib.services import base_service

_APK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data/scrcpy-server-v3.3.jar'
)
_TARGET_PATH = '/data/local/tmp/scrcpy-server.jar'

_LAUNCHING_SERVER_CMD_TEMPLATE = (
    'shell',
    'CLASSPATH=/data/local/tmp/scrcpy-server.jar',
    'app_process',
    '/',
    'com.genymobile.scrcpy.Server',
    '3.3',  # Using hardcoded version for now, as build_config is internal
    'log_level=DEBUG',
    'tunnel_forward=true',
    'stay_awake=true',
    'video=true',
    'control=false',
    'video_encoder=raw',
)

_LAUNCHING_SERVER_CMD_WITH_AUDIO = _LAUNCHING_SERVER_CMD_TEMPLATE + (
    'audio=true',
    'audio_codec=raw',
)

_LAUNCHING_SERVER_CMD_WITHOUT_AUDIO = _LAUNCHING_SERVER_CMD_TEMPLATE + (
    'audio=false',
)

_ADB_TIMEOUT_SEC = 5
_DEVICE_META_DATA_SIZE = 64
_VIDEO_META_DATA_SIZE = 64 + 4
_FRAME_HEADER_SIZE = 24
_AUDIO_HEADER_SIZE = 12
_MAX_RECV_BYTES = 30000000

_MAX_CONNECTION_ATTEMPTS = 100
_CONNECTION_WAIT_TIME_SEC = 0.1
_RETRY_TIMES = 10
_RETRY_INTERVAL_SEC = 3.0
_MAX_FPS = 30
_VIDEO_BIT_RATE = 100000
_MAX_VIDEO_SIZE = 1080
_FORWARD_PORT_RETRY_TIMES = 3
_FORWARD_PORT_RETRY_INTERVAL_SEC = 0.5
_MAX_WEAR_FPS = 10
_MAX_WEAR_FRAME_SIZE = 400

_SCRCPY_DEFAULT_AUDIO_SAMPLE_RATE = 48000
_SCRCPY_DEFAULT_AUDIO_CHANNEL_COUNT = 2
_SCRCPY_DEFAULT_AUDIO_SAMPLE_WIDTH = 2

# prevent potential race condition in port allocation and forwarding
ADB_PORT_LOCK = threading.Lock()


@enum.unique
class AudioSourceType(enum.StrEnum):
    """An enum for the scrcpy audio source."""

    MIC = 'mic'
    PLAYBACK = 'playback'
    MIC_VOICE_COMMUNICATION = 'mic-voice-communication'
    VOICE_CALL_DOWN_LINK = 'voice-call-downlink'
    OUTPUT = 'output'


@enum.unique
class CaptureOrientationType(enum.StrEnum):
    """An enum class to indicate the capture orientation type (clockwise)."""

    INITIAL_ORIENTATION = '0'  # Use the initial orientation from the device.
    ZERO_DEGREE = ' @0'  # Lock the recording orientation to 0 degree.
    NINETY_DEGREE = ' @90'  # Lock the recording orientation to 90 degree.
    ONE_EIGHTY_DEGREE = ' @180'  # Lock the recording orientation to 180 degree.
    TWO_SEVENTY_DEGREE = ' @270'  # Lock the recording orientation to 270 degree.


@dataclasses.dataclass
class Configs:
    """A configuration object for configuring the video record service.

    Attributes:
      output_dir: Path to directory to save recordings
      video_bit_rate: Bit rate to save screen recordings
      max_fps: Maximum frame per second
      max_video_size: Maximum video height
      raise_when_no_frames: whether to raise exception when no frame recorded
      display_id: Display identifier in case of multiple displays
      capture_orientation: The capture orientation type
      enable_audio: Whether to enable audio recording
      audio_source: The audio source to record
      save_audio_file: Whether to save audio file
      restart_after_create_excerpts: Whether to restart screen recording after
        creating excerpts.
    """

    output_dir: Optional[str] = None
    video_bit_rate: Optional[int] = _VIDEO_BIT_RATE
    max_fps: Optional[int] = _MAX_FPS
    max_video_size: Optional[int] = _MAX_VIDEO_SIZE
    raise_when_no_frames: Optional[bool] = True
    display_id: Optional[int] = None
    capture_orientation: Optional[CaptureOrientationType] = (
        CaptureOrientationType.INITIAL_ORIENTATION
    )
    enable_audio: bool = False
    audio_source: str = AudioSourceType.MIC
    save_audio_file: bool = False
    restart_after_create_excerpts: bool = True

    def __repr__(self) -> str:
        """Returns a string representation of the Configs object."""
        return (
            f'Configs: output_dir={self.output_dir}'
            f' video_bit_rate={self.video_bit_rate}'
            f' max_fps={self.max_fps}'
            f' max_video_size={self.max_video_size}'
            f' raise_when_no_frames={self.raise_when_no_frames}'
            f' display_id={self.display_id}'
            f' capture_orientation={self.capture_orientation}'
            f' enable_audio={self.enable_audio}'
            f' audio_source={self.audio_source}'
            f' save_audio_file={self.save_audio_file}'
            f' restart_after_create_excerpts={self.restart_after_create_excerpts}'
        )


@dataclasses.dataclass
class VideoMetadata:
    """A dataclass of metadata of the recorded video.

    Attributes:
      bit_rate: Bit rate of the video
      max_height: Configured maximum video resolution at height dimension
      width: Actual video resolution (width)
      height: Actual video resolution (height)
      fps: Frame per second
      display_id: Display identifier
      orientation: The rotation of the video
    """

    bit_rate: Optional[int] = None
    max_height: Optional[int] = None
    width: int = 0
    height: int = 0
    fps: Optional[int] = _MAX_FPS
    display_id: Optional[int] = None
    orientation: CaptureOrientationType = (
        CaptureOrientationType.INITIAL_ORIENTATION
    )


@dataclasses.dataclass
class ServiceTimeStamps:
    """A dataclass of timestamps of the screen record service.

    Attributes:
      time_norm: The number to divide to normalize scrcpy
        timestamp to time in seconds
      start_time: Host time when receives the first frame
      last_time: Host time when receives last frame
      end_time: Host time when the service stops
      first_frame_time: Device timestamp of the first video frame
      last_frame_time: Device timestamp of last video frame
      use_frame_time: Whether to use device frame timestamp for video encoding,
          if False, then use host time and re-align the device timestamp
    """

    time_norm: Optional[float] = 1e9
    start_time: float = 0
    last_time: float = 0
    end_time: float = 0
    first_frame_time: float = 0
    last_frame_time: float = 0
    use_frame_time: Optional[bool] = True


class Error(errors.ServiceError):
    """Error type for ScreenRecorder service."""

    SERVICE_TYPE = 'ScreenRecorder'


class ScreenRecorder(base_service.BaseService):
    """A service for recording screen videos from an Android device.

    This service is implemented with `scrcpy` dependency.
    To start the screen record service, we push the `scrcpy-server` apk and run
    the server on the device. The host acts as a client and connect to the
    `scrcpy` server with sockets. Then the host receives video frames from the
    `scrcpy` server and write to a video file.
    """

    def __init__(
        self,
        device: android_device.AndroidDevice,
        configs: Optional[Configs] = None,
    ) -> None:
        """Initializes a ScreenRecorder instance.

        Args:
            device: The AndroidDevice object to record from.
            configs: A Configs object with recording parameters.
        """
        if configs is None:
            configs = Configs()
        super().__init__(device, configs)
        device.log.debug('Initializing screen recorder with %s.', repr(configs))
        self._device = device
        self._server_proc = None
        # `_prepared` is set to true when `_setup()` is called, where the scrcpy
        # server starts and socket connection is built. It means the service is
        # ready for recording the video
        self._prepared = False
        self._is_alive = False
        self.port = None
        self._video_socket = None
        self._audio_socket = None
        self.output_dir = configs.output_dir or device.log_path
        # output filename for the final uploaded video file, combined from
        # temporary video file and audio file if enabled audio recording
        self._final_output_filename = None
        # filename for the temporary video file, will be merged at last
        self._temp_video_file = None
        # filename for the temporary audio file, will be merged at last
        self._temp_audio_file = None

        self._raise_when_no_frames = configs.raise_when_no_frames
        self._video_meta = VideoMetadata(
            bit_rate=configs.video_bit_rate or _VIDEO_BIT_RATE,
            max_height=configs.max_video_size or _MAX_VIDEO_SIZE,
            width=0,
            height=0,
            fps=configs.max_fps or _MAX_FPS,
            display_id=configs.display_id,
            orientation=configs.capture_orientation
            or CaptureOrientationType.INITIAL_ORIENTATION,
        )
        self._timestamps = ServiceTimeStamps()
        self._last_frame = None
        self._video_writer = None
        self._audio_writer = None
        # True when it's the first time to call `start()` and `stop()`
        self._first_run = True
        self._video_files = []  # released video files, will be merged at last
        self._video_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._video_job = None
        self._audio_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._audio_job = None
        self._enable_audio = configs.enable_audio
        if self._enable_audio:
            self._check_ffmpeg_availability()
        self._audio_source = configs.audio_source or AudioSourceType.MIC
        self._save_audio_file = configs.save_audio_file or False
        self._restart_after_create_excerpts = (
            configs.restart_after_create_excerpts
        )

    def __repr__(self) -> str:
        """Returns a string representation of the ScreenRecorder object."""
        return (
            f'ScreenRecorder(serial={self._device.serial}'
            f' dir={self.output_dir} bit_rate={self._video_meta.bit_rate}'
            f' max_height={self._video_meta.max_height})'
        )

    @property
    def is_alive(self) -> bool:
        """True if the service is recording video; False otherwise."""
        return self._is_alive

    def _setup(self) -> None:
        """Prepares the service for recording.

        1. Uploads the `scrcpy-server` apk to the device.
        2. Forwards the `scrcpy` server port to the localhost port in config
           for later socket connection.
        3. Starts the `scrcpy` server on the device.
        """
        self._device.adb.push(
            [_APK_PATH, _TARGET_PATH], timeout=_ADB_TIMEOUT_SEC
        )
        self._forward_port()
        self._start_server()
        self._start_video_connection()
        if self._enable_audio:
            self._start_audio_connection()
        self._read_metadata()
        if self._video_writer is None:
            self._set_video_writer()
        if self._audio_writer is None and self._enable_audio:
            self._set_audio_writer()

        if not self._first_run:
            # Pad blank frames for the duration when the service is stopped
            host_time = time.monotonic()
            timestamp = self._get_fake_timestamp(host_time)
            self._add_frame(timestamp, host_time, b'')
        self._prepared = True

    def start(self) -> None:
        """Starts the screen recording service."""
        if self._is_alive:
            return
        self._is_alive = True
        # The `_video_writer` will be set in `_setup()`
        # get the state before the function is called
        log_start_flag = self._video_writer is None
        if not self._prepared:
            self._setup()
        # Re-initialize executors if they have been shut down
        if self._video_executor is None:
            self._video_executor = futures.ThreadPoolExecutor(max_workers=1)
        if self._enable_audio and self._audio_executor is None:
            self._audio_executor = futures.ThreadPoolExecutor(max_workers=1)

        # when start() and stop() are called multiple times during each test, only
        # log first start time.
        if log_start_flag and self._final_output_filename:
            self._device.log.info(
                self._generate_video_start_log_with_filename(
                    self._device, self._final_output_filename
                )
            )
        self._video_job = self._video_executor.submit(self._video_stream_loop)
        if self._enable_audio:
            self._audio_job = self._audio_executor.submit(
                self._audio_stream_loop
            )

    def cleanup(self) -> None:
        """Cleans up the screen recording service.

        1. Cancels port forwarding.
        2. Deletes the `scrcpy-server` apk from the device.
        3. Kills the `scrcpy` server.
        4. Closes the sockets.
        """
        self._prepared = False
        try:  # cancels port forwarding
            self._device.adb.forward(['--remove', f'tcp:{str(self.port)}'])
            self._device.adb.shell(['rm', '-f', _TARGET_PATH])
        except adb.AdbError:
            # happens when device reboots, can be ignored
            self._device.log.debug(
                'Failed to cancel port forwarding or delete apk file.'
            )
        self._kill_scrcpy_server()
        self._video_executor.shutdown(wait=False)
        self._audio_executor.shutdown(wait=False)
        self._video_executor = None
        self._audio_executor = None

        video_result = None
        if self._video_job:
            exception = self._video_job.exception()
            if exception:
                raise exception
            video_result = self._video_job.result()
        self._device.log.debug(
            'Screen record service video loop thread returns'
            f' with result={video_result}'
        )
        self._video_job = None

        if self._enable_audio:
            audio_result = None
            if self._audio_job:
                exception = self._audio_job.exception()
                if exception:
                    raise exception
                audio_result = self._audio_job.result()
            self._device.log.debug(
                'Screen record service audio loop thread returns'
                f' with result={audio_result}'
            )
            self._audio_job = None

        if self._video_socket is not None:
            self._video_socket.close()
            self._video_socket = None
        if self._audio_socket is not None:
            self._audio_socket.close()
            self._audio_socket = None

    def stop(self) -> None:
        """Stops the screen recording service and do clean up."""
        # The `stop` function will not release the video file. Users should call
        # `create_output_excerpts` to get the generated file (typically at the end
        # of: `setup_class`, `teardown_test` or `teardown_class`.
        # If no `create_output_excerpts` is called, the video file is supposed to be
        # released automatically when the program exits.
        if not self._is_alive:
            return
        self._is_alive = False
        self.cleanup()
        # set end timestamp and add blank frame
        self._timestamps.end_time = time.monotonic()
        timestamp = self._get_fake_timestamp(self._timestamps.end_time)
        padding_nums = self._add_frame(
            timestamp, self._timestamps.end_time, b''
        )
        self._device.log.debug(f'Padding {padding_nums} frames at stop time.')
        self._timestamps.use_frame_time = False
        self._first_run = False

    def _align_time(self, timestamp: float, curtime: float) -> None:
        """Aligns host time and device timestamp."""
        self._device.log.debug(
            f'Aligning time: host={curtime} device={timestamp}'
        )

        # (Host) curtime - starttime ~= (Device) timestamp - first_frame_time
        # (Host) curtime - lasttime ~= (Device) timestamp - last_frame_time
        timediff = timestamp - curtime
        self._timestamps.first_frame_time = (
            timediff + self._timestamps.start_time
        )
        self._timestamps.last_frame_time = timediff + self._timestamps.last_time

    def _get_fake_timestamp(self, host_time: float) -> float:
        """Gets a fake device timestamp."""
        # Use host time and previous device timestamp
        # to generate a fake device timestamp.
        # T_device_1 ~= T_device_0 + (T_host_1 - T_host_0)
        timestamp = host_time - self._timestamps.last_time
        timestamp = timestamp + self._timestamps.last_frame_time
        return timestamp

    def _release_final_output_video_file(self) -> None:
        """Processes frames for both video and audio and writes final output video file.

        Raises:
            Error: Raise if no frame recorded.
        """
        if self._last_frame is None:
            self._device.log.debug('No frame found in screenrecord service.')
            if self._raise_when_no_frames:
                raise Error(
                    self._device, 'No frame found in screenrecord service.'
                )
            return
        self._timestamps.end_time = time.monotonic()
        timestamp = self._get_fake_timestamp(self._timestamps.end_time)
        padding_nums = self._add_frame(
            timestamp, self._timestamps.end_time, b''
        )
        self._device.log.debug(
            f'Padding {padding_nums} frames at release time.'
        )

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            if self._audio_writer is not None and self._enable_audio:
                self._audio_writer.close()
                self._audio_writer = None
                # Combine video and audio files to the final output file.
                self._combine_video_audio_files()
            last_video = os.path.join(
                self.output_dir, self._final_output_filename
            )
            self._video_files.append(last_video)

    def _combine_video_audio_files(self) -> None:
        """Combines video and audio files to the final output file."""
        if (
            not self._enable_audio
            or self._temp_video_file is None
            or self._temp_audio_file is None
        ):
            raise Error(
                self._device,
                'Audio recording is not enabled or some temporary files are'
                ' missing. Cannot combine video and audio files.',
            )
        self._check_ffmpeg_availability()
        video_file_path = os.path.join(
            self.output_dir, self._temp_video_file
        )
        audio_file_path = os.path.join(
            self.output_dir, self._temp_audio_file
        )
        output_file_path = os.path.join(
            self.output_dir, self._final_output_filename
        )

        ffmpeg_cmd = [
            'ffmpeg',
            '-i',
            video_file_path,
            '-i',
            audio_file_path,
            '-c:v',
            'copy',
            '-c:a',
            'aac',
            '-strict',
            '-2',
            output_file_path,
        ]
        self._device.log.debug(
            f'Merging video and audio with ffmpeg: {" ".join(ffmpeg_cmd)}'
        )
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Error(
                self._device,
                f'FFmpeg merging failed with error: {e.stderr.decode()}',
            ) from e

    def _add_frame(
        self,
        timestamp: float,
        host_time: float,
        frame_bytes: bytes,
    ) -> int:
        """Adds a frame and write the previous frame to the video file.

        Args:
            timestamp: float, the device timestamp in seconds of this frame.
            host_time: float, the host timestamp in seconds of this frame.
            frame_bytes: bytes, an image to be decoded. Indicates blank frame if
                set to b''.

        Returns:
            frame_num: int, the number of frames written in this call.
        When we add a frame, we duplicate the previous frame and write it to the
        video file.
        """
        if self._timestamps.start_time == 0:
            self._timestamps.start_time = host_time

        frame_num = 0
        if self._timestamps.first_frame_time == 0:
            self._timestamps.first_frame_time = timestamp
        else:
            # duplicate the last frame
            frame_num = round(
                (timestamp - self._timestamps.last_frame_time)
                * self._video_meta.fps
            )
            for num in range(frame_num):
                if self._video_writer is not None:
                    self._write_frame_to_file(self._last_frame)
                else:
                    # No padding frames at release time.
                    frame_num = num
                    break

        self._last_frame = frame_bytes
        self._timestamps.last_frame_time = timestamp
        self._timestamps.last_time = host_time
        return frame_num

    @retrying.retry(
        stop_max_attempt_number=_RETRY_TIMES,
        wait_fixed=_RETRY_INTERVAL_SEC,
        retry_on_exception=lambda e: isinstance(
            e, (adb.AdbError, ConnectionError)
        ),
    )
    def _restart(self) -> None:
        """Restarts the server.

        Called when the socket is disconnected.
        """
        # Stop trying to restart server when we are stopping the service.
        if not self._is_alive:
            return
        self._device.log.debug('Restarting screen record service.')
        self._prepared = False
        if self._video_socket:
            self._video_socket.close()
            self._video_socket = None
        if self._audio_socket:
            self._audio_socket.close()
            self._audio_socket = None
        if self._is_alive:
            self._kill_scrcpy_server()
            self._setup()

    def _recv_bytes(
        self,
        stream_socket: socket.socket | None,
        k_bytes: int,
    ) -> bytes:
        """Wraps `recv` function to read k bytes from socket.

        Args:
            stream_socket: socket, a socket object.
            k_bytes: int, the number of bytes to read.

        Returns:
            ret_data: bytes, empty when the socket is closed.
        """
        ret_data = bytes()
        bytes_read = 0
        if stream_socket is None:
            self._device.log.debug(
                'Stream socket is None when receiving bytes, return empty'
                ' bytes.'
            )
            return bytes()
        while bytes_read < k_bytes:
            try:
                data = stream_socket.recv(k_bytes - bytes_read)
                if not data:
                    return bytes()
            except socket.timeout as e:
                self._device.log.debug(
                    'Server side got lost unexpectedly, socket timeout: %s,'
                    ' return empty bytes.',
                    e,
                )
                return bytes()
            ret_data += data
            bytes_read += len(data)
        return ret_data

    def _handle_socket_disconnection(self) -> None:
        """Handles socket disconnection when reading frames.

        1. Inserts a blank frame.
        2. Sets `use_frame_time` to False, which indicates that we should re-align
           the stored frame timestamps after we connect to a restarted server.
        3. Restarts the server.
        """
        # do not insert blank frame if there is no previous frame
        if self._timestamps.last_frame_time != 0:
            host_time = time.monotonic()
            timestamp = self._get_fake_timestamp(host_time)
            self._add_frame(timestamp, host_time, b'')
        self._timestamps.use_frame_time = False
        if self._is_alive:
            self._restart()

    def _video_stream_loop(self) -> None:
        """Receives video frames through video socket."""
        while self._is_alive:
            header = self._recv_bytes(self._video_socket, _FRAME_HEADER_SIZE)
            if not header:
                # This is expected to happen when socket connection is closed.
                # It happens when the server is killed unexpectedly or the adb
                #    is disconnected, which should be rare.
                self._device.log.debug(
                    'No video frame header received, '
                    'restarting device-side server and reconnect'
                )
                self._handle_socket_disconnection()
                continue
            timestamp = int.from_bytes(header[12:20], 'big', signed=True)
            framesize = int.from_bytes(header[20:24], 'big', signed=False)
            host_frame_time = time.monotonic()

            if framesize > _MAX_RECV_BYTES:
                raise Error(
                    self._device, f'Frame size {framesize} is too large.'
                )

            data = self._recv_bytes(self._video_socket, framesize)
            if not data:
                self._device.log.debug(
                    'LOOP: Not receiving video frame data.'
                    ' Restart server and reconnect.'
                )
                self._handle_socket_disconnection()
                continue
            # timestamps between different server run is not reliable
            # we should reset the frame timestamp of the previous run
            frame_timestamp = timestamp / self._timestamps.time_norm
            if not self._timestamps.use_frame_time:
                self._align_time(frame_timestamp, host_frame_time)
                self._timestamps.use_frame_time = True
            self._add_frame(frame_timestamp, host_frame_time, data)

    def _audio_stream_loop(self) -> None:
        """Receives audio data through audio sockets."""
        while self._is_alive:
            header = self._recv_bytes(self._audio_socket, _AUDIO_HEADER_SIZE)
            if not header:
                # This is expected to happen when socket connection is closed.
                # It happens when the server is killed unexpectedly or the adb
                #    is disconnected, which should be rare.
                self._device.log.debug(
                    'No audio frame header received, '
                    'restarting device-side server and reconnect'
                )
                self._handle_socket_disconnection()
                continue

            framesize = int.from_bytes(header[8:12], 'big', signed=False)
            if framesize > _MAX_RECV_BYTES:
                raise Error(
                    self._device, f'Frame size {framesize} is too large.'
                )

            data = self._recv_bytes(self._audio_socket, framesize)
            if not data:
                self._device.log.debug(
                    'LOOP: Not receiving audio frame data.'
                    ' Restart server and reconnect.'
                )
                self._handle_socket_disconnection()
                continue
            self._write_audio_data_to_file(data)

    def _start_video_connection(self) -> None:
        """Starts the video connection to the server and sets sockets."""
        self._device.log.debug('Starting new scrcpy video socket connection')
        for attempts in range(_MAX_CONNECTION_ATTEMPTS):
            try:
                self._video_socket = self._create_socket_connection(attempts)

                dummy_byte = self._recv_bytes(self._video_socket, 1)
                if dummy_byte:
                    break
                raise ConnectionRefusedError
            except (
                socket.error,
                ConnectionAbortedError,
                ConnectionResetError,
                ConnectionRefusedError,
            ):
                if self._video_socket:
                    self._video_socket.close()
                    self._video_socket = None
                time.sleep(_CONNECTION_WAIT_TIME_SEC)
        else:
            self._device.log.debug('Video connection failed.')
            raise ConnectionError(
                'Failed to build video socket connection with scrcpy server.'
            )

    def _start_audio_connection(self) -> None:
        """Starts the audio connection to the server and sets sockets."""
        self._device.log.debug('Starting new scrcpy audio socket connection')
        for attempts in range(_MAX_CONNECTION_ATTEMPTS):
            try:
                self._audio_socket = self._create_socket_connection(attempts)

                dummy_byte = self._audio_socket.recv(4)
                if dummy_byte:
                    self._device.log.debug(
                        f'Audio socket connected with codec id: {dummy_byte}'
                    )
                    break
                raise ConnectionRefusedError
            except (
                socket.error,
                ConnectionAbortedError,
                ConnectionResetError,
                ConnectionRefusedError,
            ):
                if self._audio_socket:
                    self._audio_socket.close()
                    self._audio_socket = None
                time.sleep(_CONNECTION_WAIT_TIME_SEC)
        else:
            self._device.log.debug('Audio connection failed.')
            raise ConnectionError(
                'Failed to build audio socket connection with scrcpy server.'
            )

    def _create_socket_connection(self, attempts: int) -> socket.socket:
        """Creates a socket connection to the server."""
        socket_type = socket.AF_INET if attempts % 2 == 0 else socket.AF_INET6

        create_socket = socket.socket(socket_type, socket.SOCK_STREAM)
        create_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        create_socket.connect_ex(('localhost', self.port))
        return create_socket

    def _read_metadata(self) -> None:
        """Reads device name and screen size through socket."""
        if self._video_socket is None and self._is_alive:
            raise ValueError(
                '_start_connection() not call, _video_socket is None.'
            )

        device_metadata = self._recv_bytes(
            self._video_socket, _DEVICE_META_DATA_SIZE
        )
        self._device.log.debug(
            f'Socket recv device metadata: {device_metadata}'
        )
        if not device_metadata and self._is_alive:
            raise ConnectionError('Did not receive device metadata!')
        device_name = device_metadata[0:64].decode('utf-8')
        self._device.log.debug(f'Device name: {device_name}')

        video_metadata = self._recv_bytes(
            self._video_socket, _VIDEO_META_DATA_SIZE
        )
        self._device.log.debug(
            f'Socket recv video metadata: {video_metadata}'
        )
        if not video_metadata and self._is_alive:
            raise ConnectionError('Did not receive video metadata!')
        self._video_meta.width = int.from_bytes(
            video_metadata[64:66], 'big', signed=True
        )
        self._video_meta.height = int.from_bytes(
            video_metadata[66:68], 'big', signed=True
        )
        self._device.log.debug(
            f'WxH: {self._video_meta.width}x{self._video_meta.height}',
        )
        # For wearable form factor, we need to set the fps to 10 to save resource.
        if (
            self._video_meta.width < _MAX_WEAR_FRAME_SIZE
            and self._video_meta.height < _MAX_WEAR_FRAME_SIZE
            and self._video_meta.fps == _MAX_FPS
        ):
            self._video_meta.fps = _MAX_WEAR_FPS

    def _set_video_writer(self) -> None:
        """Sets up the video writer."""
        # Video writer with OpenCV pyclif package
        filename = self._device.generate_filename(
            'video', extension_name='mp4'
        )
        self._final_output_filename = filename
        # If audio is enabled, we will write the video to a temporary file.
        if self._enable_audio:
            filename = self._device.generate_filename(
                'only_video', extension_name='mp4'
            )
            self._temp_video_file = filename
        self._video_writer = cv2.VideoWriter(
            os.path.join(self.output_dir, filename),
            cv2.VideoWriter_fourcc(*'H264'),
            self._video_meta.fps,
            frameSize=(
                self._video_meta.width,
                self._video_meta.height,
            ),
        )

    def _set_audio_writer(self) -> None:
        """Sets up the audio writer."""
        # Audio writer with wave package
        filename = self._device.generate_filename(
            'only_audio', extension_name='wav'
        )
        self._temp_audio_file = filename
        self._audio_writer = wave.open(
            os.path.join(self.output_dir, filename), 'wb'
        )
        self._audio_writer.setnchannels(_SCRCPY_DEFAULT_AUDIO_CHANNEL_COUNT)
        self._audio_writer.setsampwidth(_SCRCPY_DEFAULT_AUDIO_SAMPLE_WIDTH)
        self._audio_writer.setframerate(_SCRCPY_DEFAULT_AUDIO_SAMPLE_RATE)

    def _start_server(self) -> None:
        """Starts the scrcpy server on the device."""
        server_cmd = [adb.ADB, '-s', str(self._device.serial)]
        if self._enable_audio:
            server_cmd += list(_LAUNCHING_SERVER_CMD_WITH_AUDIO)
            server_cmd.append(f'audio_source={self._audio_source}')
        else:
            server_cmd += list(_LAUNCHING_SERVER_CMD_WITHOUT_AUDIO)

        if self._video_meta.bit_rate:
            server_cmd.append(f'video_bit_rate={self._video_meta.bit_rate}')
        if self._video_meta.fps:
            server_cmd.append(f'max_fps={self._video_meta.fps}')
        if self._video_meta.max_height:
            server_cmd.append(f'max_size={self._video_meta.max_height}')
        if self._video_meta.display_id is not None:
            server_cmd.append(f'display_id={self._video_meta.display_id}')
        server_cmd.append(
            f'capture_orientation={self._video_meta.orientation}'
        )
        cmd_str = ' '.join(server_cmd)
        self._device.log.debug(
            f'Starting server with Popen command: {cmd_str}'
        )
        self._server_proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self._server_proc.stdout is not None
        stdout_line = b''
        for output_line in iter(self._server_proc.stdout.readline, ''):
            if isinstance(output_line, bytes):
                stdout_line = output_line.decode('utf-8').strip()
            else:
                stdout_line = str(output_line).strip()
            if stdout_line.find('[server] INFO: Device: ') == 0:
                break
            # First line wasn't the device info line (most likely aborted)
            self._device.log.debug(
                'Server output line has no device info: Server stdout:'
                f' {stdout_line}'
            )
            if not self._is_alive:
                self._device.log.debug(
                    'The service is stopped, skip raising error.'
                )
                return
            raise Error(
                self._device, f'Start server failed. stdout: {stdout_line}'
            )
        self._device.log.debug(f'Server started: {stdout_line}.')

    def _kill_scrcpy_server(self) -> None:
        """Kills the scrcpy server process."""
        if self._server_proc:
            self._device.log.debug('Killing server process.')
            self._server_proc.kill()
            self._server_proc = None
        try:
            self._device.adb.shell(['pkill', '-f', 'scrcpy'])
        except adb.AdbError:
            pass  # Expected if no scrcpy processes are running.

    def _write_frame_to_file(self, frame_bytes: bytes) -> None:
        """Writes a frame to the video file."""
        if self._video_writer is None:
            raise ValueError(
                '_set_video_writer() not call, _video_writer is None.'
            )
        # if frame_bytes is empty, write a blank frame.
        frame = (
            cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if frame_bytes
            else np.zeros(
                (self._video_meta.height, self._video_meta.width, 3), np.uint8
            )
        )
        self._video_writer.write(frame)

    def _write_audio_data_to_file(self, frame: bytes) -> None:
        """Writes audio data to the audio file."""
        if self._audio_writer is None:
            raise ValueError(
                '_set_audio_writer() not call, _audio_writer is None.'
            )
        try:
            self._audio_writer.writeframes(frame)
        except wave.Error:
            self._device.log.exception('Failed to write non-empty frame.')

    def create_output_excerpts(
        self,
        test_info: runtime_test_info.RuntimeTestInfo,
    ) -> List[Any]:
        """Creates excerpts for the videos from the recording session.

        Args:
            test_info: The currently running test to use for determining the
                excerpt storage location.

        Returns:
            The list of location of the saved video file.
        Raises:
            Error: Raise if the video file is not found.
        """
        if self._is_alive:
            self.stop()
        self._release_final_output_video_file()
        if len(self._video_files) == 1:
            output_excerpts = []
            file_location = self._video_files[-1]
            target_location = os.path.join(
                test_info.output_path, self._final_output_filename
            )
            self._device.log.debug(f'Save video file to {target_location}')
            shutil.move(file_location, target_location)
            output_excerpts.append(target_location)
            # Remove temporary files if audio recording is enabled.
            if self._enable_audio:
                self._device.log.debug('Removing temporary files.')
                try:
                    for file in (
                        self._temp_video_file,
                        self._temp_audio_file,
                    ):
                        file_path = os.path.join(self.output_dir, file)
                        # Keep audio file if save_audio_file is configured to True.
                        if (
                            file == self._temp_audio_file
                            and self._save_audio_file
                        ):
                            audio_target_location = os.path.join(
                                test_info.output_path, self._temp_audio_file
                            )
                            shutil.move(file_path, audio_target_location)
                            output_excerpts.append(audio_target_location)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                except OSError as e:
                    self._device.log.debug(
                        f'Error removing temporary files: {e}'
                    )
            self._video_files = []
            if self._restart_after_create_excerpts:
                self.start()
            return output_excerpts
        self._device.log.info('Service writes none or more than one videos.')
        raise Error(
            self._device, 'Service writes none or more than one videos.'
        )

    @retrying.retry(
        stop_max_attempt_number=_FORWARD_PORT_RETRY_TIMES,
        wait_fixed=_FORWARD_PORT_RETRY_INTERVAL_SEC,
        retry_on_exception=lambda e: isinstance(e, (adb.AdbError)),
    )
    def _forward_port(self) -> None:
        """Forwards the device port to a local port."""
        with ADB_PORT_LOCK:  # lock when allocating and forwarding port
            port = self._device.adb.forward(
                ['tcp:0', 'localabstract:scrcpy']
            )
            self.port = int(port.split(b'\n')[0])

    def _generate_video_start_log_with_filename(
        self,
        device: android_device.AndroidDevice,
        filename: str,
    ) -> str:
        """Gets the log when video record start.

        Args:
            device: The Android device which records the video.
            filename: The file name which the video record is saved as.

        Returns:
            The log contains video starting time.
        """
        timestamp_str = (
            device.adb.shell(
                'echo $(date +%Y-%m-%dT%T)${EPOCHREALTIME:10:4}'
            )
            .replace(b'\n', b'')
            .decode('utf-8')
        )
        return (
            f'INFO:{device.serial} Start video recording '
            f'{timestamp_str}, '
            f'output filename {filename}'
        )

    def _check_ffmpeg_availability(self) -> None:
        """Checks if ffmpeg is installed and available in PATH."""
        if shutil.which('ffmpeg') is None:
            raise Error(
                self._device,
                (
                    'ffmpeg not found in system PATH. Please install ffmpeg to'
                    ' merge audio and video files.'
                ),
            )
