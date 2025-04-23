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

from concurrent import futures
import dataclasses
import enum
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from typing import Any, List, Optional
import cv2

from mobly import runtime_test_info
from mobly.controllers import android_device
from mobly.controllers.android_device_lib import adb
from mobly.controllers.android_device_lib import errors
from mobly.controllers.android_device_lib.services import base_service
import numpy as np
import retrying

_APK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data/scrcpy-server-v3.jar'
)
_TARGET_PATH = '/data/local/tmp/scrcpy-server.jar'
_LAUNCHING_SERVER_CMD = (
    'shell',
    'CLASSPATH=/data/local/tmp/scrcpy-server.jar',
    'app_process',
    '/',
    'com.genymobile.scrcpy.Server',
    '3.1',
    'log_level=DEBUG',
    'tunnel_forward=true',
    'stay_awake=true',
    'audio=false',
    'video_codec_options=image_format:string=JPEG',
    'video_encoder=raw',
    'send_device_meta=true',
    'send_frame_meta=true',
    'send_dummy_byte=true',
    'send_codec_meta=true',
    'control=false',
)

_ADB_TIMEOUT_SEC = 5
_META_DATA_SIZE = 64 + 4
_FRAME_HEADER_SIZE = 24
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

# prevent potential race condition in port allocation and forwarding
ADB_PORT_LOCK = threading.Lock()


@enum.unique
class CaptureOrientationType(enum.StrEnum):
  """An enum class to indicate the capture orientation type (clockwise)."""

  INITIAL_ORIENTATION = '@'
  ZERO_DEGREE = '@0'
  NINETY_DEGREE = '@90'
  ONE_EIGHTY_DEGREE = '@180'
  TWO_SEVENTY_DEGREE = '@270'
  

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

  def __repr__(self) -> str:
    return (
        f'Configs(dir={self.output_dir}'
        f' bit_rate={self.video_bit_rate}'
        f' max_size={self.max_video_size})'
        f' max_fps={self.max_fps}'
        f' raise_when_no_frames={self.raise_when_no_frames}'
        f' display_id={self.display_id}'
        f' capture_orientation={self.capture_orientation}'
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

  def __init__(self,
               device: android_device.AndroidDevice,
               configs: Optional[Configs] = None) -> None:
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
    self.output_dir = configs.output_dir or device.log_path
    self._output_filename = None
    self._raise_when_no_frames = configs.raise_when_no_frames
    self._video_meta = VideoMetadata(
        bit_rate=configs.video_bit_rate or _VIDEO_BIT_RATE,
        max_height=configs.max_video_size or _MAX_VIDEO_SIZE,
        width=0,
        height=0,
        fps=configs.max_fps or _MAX_FPS,
        display_id=configs.display_id,
        orientation=configs.capture_orientation,
    )
    self._timestamps = ServiceTimeStamps()
    self._last_frame = None
    self._video_writer = None
    # True when it's the first time to call `start()` and `stop()`
    self._first_run = True
    self._video_files = []    # released video files, will be merged at last
    self._executor = futures.ThreadPoolExecutor(max_workers=1)
    self._job = None

  def __repr__(self) -> str:
    return (f'ScreenRecorder(serial={self._device.serial}'
            f' dir={self.output_dir} bit_rate={self._video_meta.bit_rate}'
            f' max_height={self._video_meta.max_height})')

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
    self._device.adb.push([_APK_PATH, _TARGET_PATH], timeout=_ADB_TIMEOUT_SEC)
    self._forward_port()
    self._start_server()
    self._start_connection()
    self._read_metadata()
    if self._video_writer is None:
      self._set_writer()
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
    log_start_flag = True if self._video_writer is None else False
    if not self._prepared:
      self._setup()
    # when stop() is called, the _executor will be shutdown
    # we need to reset an executor
    self._executor = futures.ThreadPoolExecutor(max_workers=1)
    # when start() and stop() are called multiple times during each test, only
    # log first start time.
    if log_start_flag and self._output_filename:
      self._device.log.info(
          self._generate_video_start_log_with_filename(
              self._device, self._output_filename
          )
      )
    self._job = self._executor.submit(self._stream_loop)

  def cleanup(self) -> None:
    """Cleanup.

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
          'Failed to cancel port forwarding or delete apk file.')
    self._kill_scrcpy_server()
    self._executor.shutdown(wait=False)

    result = None
    if self._job:
      exception = self._job.exception()
      if exception:
        raise exception
      result = self._job.result()
    self._device.log.debug(f'Screen record service loop thread returns'
                           f' with result={result}')
    self._job = None

    if self._video_socket is not None:
      self._video_socket.close()
      self._video_socket = None

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
    padding_nums = self._add_frame(timestamp, self._timestamps.end_time, b'')
    self._device.log.debug(f'Padding {padding_nums} frames at stop time.')
    self._timestamps.use_frame_time = False
    self._first_run = False

  def _align_time(self, timestamp: float, curtime: float) -> None:
    """Align host time and device timestamp."""
    self._device.log.debug(f'Aligning time: host={curtime} device={timestamp}')

    # (Host) curtime - starttime ~= (Device) timestamp - first_frame_time
    # (Host) curtime - lasttime ~= (Device) timestamp - last_frame_time
    timediff = timestamp - curtime
    self._timestamps.first_frame_time = timediff + self._timestamps.start_time
    self._timestamps.last_frame_time = timediff + self._timestamps.last_time

  def _get_fake_timestamp(self, host_time: float) -> float:
    """Get a fake device timestamp."""
    # Use host time and previous device timestamp
    # to generate a fake device timestamp.
    # T_device_1 ~= T_device_0 + (T_host_1 - T_host_0)
    timestamp = host_time - self._timestamps.last_time
    timestamp = timestamp + self._timestamps.last_frame_time
    return timestamp

  def _release_video_file(self) -> None:
    """Processes frames and writes video file.

    Raises:
      Error: Raise if no frame recorded.
    """
    if self._last_frame is None:
      self._device.log.debug('No frame found in screenrecord service.')
      if self._raise_when_no_frames:
        raise Error(self._device, 'No frame found in screenrecord service.')
      else:
        return
    self._timestamps.end_time = time.monotonic()
    timestamp = self._get_fake_timestamp(self._timestamps.end_time)
    padding_nums = self._add_frame(timestamp, self._timestamps.end_time, b'')
    self._device.log.debug(f'Padding {padding_nums} frames at release time.')

    if self._video_writer is not None:
      self._video_writer.release()
      self._video_writer = None
      last_video = os.path.join(self.output_dir, self._output_filename)
      self._video_files.append(last_video)

  def _add_frame(
      self, timestamp: float, host_time: float, frame_bytes: bytes
  ) -> int:
    """Adds a frame and write the previous frame to the video file.

    Args:
      timestamp: float, the device timestamp in seconds of this frame.
      host_time: float, the host timestamp in seconds of this frame.
      frame_bytes: bytes, an image to be decoded. Indicates blank frame if set
        to b''.

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
      frame_num = round((timestamp - self._timestamps.last_frame_time) *
                        self._video_meta.fps)
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
    if self._is_alive:
      self._kill_scrcpy_server()
      self._setup()
    return

  def _recv_bytes(self, k_bytes: int) -> bytes:
    """Wrap `recv` function to read k bytes from socket.

    Args:
      k_bytes: int
    Returns:
      ret_data: bytes, empty when the socket is closed.
    """
    ret_data = bytes()
    bytes_read = 0
    if self._video_socket is None:
      raise ValueError('_start_connection() not call, _video_socket is None.')
    while bytes_read < k_bytes:
      data = self._video_socket.recv(k_bytes - bytes_read)
      if not data:
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

  def _stream_loop(self) -> None:
    """Receive frames through sockets."""
    while self._is_alive:
      header = self._recv_bytes(_FRAME_HEADER_SIZE)
      if not header:
        # This is expected to happen when socket connection is closed.
        # It happens when the server is killed unexpectedly or the adb
        #    is disconnected, which should be rare.
        self._device.log.debug('No frame header received, '
                               'restarting device-side server and reconnect')
        self._handle_socket_disconnection()
        continue
      timestamp = int.from_bytes(header[12:20], 'big', signed=True)
      framesize = int.from_bytes(header[20:24], 'big', signed=False)
      host_frame_time = time.monotonic()

      if framesize > _MAX_RECV_BYTES:
        raise Error(self._device, f'Frame size {framesize} is too large.')

      data = self._recv_bytes(framesize)
      if not data:
        self._device.log.debug(('LOOP: Not receiving frame data.'
                                ' Restart server and reconnect.'))
        self._handle_socket_disconnection()
        continue
      # timestamps between different server run is not reliable
      # we should reset the frame timestamp of the previous run
      frame_timestamp = timestamp / self._timestamps.time_norm
      if not self._timestamps.use_frame_time:
        self._align_time(frame_timestamp, host_frame_time)
        self._timestamps.use_frame_time = True
      self._add_frame(frame_timestamp, host_frame_time, data)

  def _start_connection(self) -> None:
    """Starts the connection to the server and sets sockets."""
    self._device.log.debug('Starting new scrcpy video socket connection')
    for attempts in range(_MAX_CONNECTION_ATTEMPTS):
      try:
        socket_type = socket.AF_INET if attempts % 2 == 0 else socket.AF_INET6
        self._video_socket = socket.socket(socket_type, socket.SOCK_STREAM)
        self._video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._video_socket.connect_ex(('localhost', self.port))

        dummy_byte = self._video_socket.recv(1)
        if dummy_byte:
          break
        else:
          raise ConnectionRefusedError
      except (socket.error, ConnectionAbortedError, ConnectionResetError,
              ConnectionRefusedError):
        if self._video_socket:
          self._video_socket.close()
          self._video_socket = None
        time.sleep(_CONNECTION_WAIT_TIME_SEC)
    else:
      self._device.log.debug('Connection failed.')
      raise ConnectionError(
          'Failed to build video socket connection with scrcpy server.'
      )

  def _read_metadata(self) -> None:
    """Reads device name and screen size through socket."""
    if self._video_socket is None:
      raise ValueError('_start_connection() not call, _video_socket is None.')
    metadata = self._video_socket.recv(_META_DATA_SIZE)
    if not metadata:
      raise ConnectionError('Did not receive metaData!')
    device_name = metadata[0:64].decode('utf-8')
    self._device.log.debug(f'Device name: {device_name}')
    self._video_meta.width = int.from_bytes(metadata[64:66], 'big', signed=True)
    self._video_meta.height = int.from_bytes(
        metadata[66:68], 'big', signed=True)
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

  def _set_writer(self) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = self._device.generate_filename('video', extension_name='mp4')
    self._output_filename = filename
    self._video_writer = cv2.VideoWriter(
        os.path.join(self.output_dir, filename),
        fourcc,
        self._video_meta.fps,
        frameSize=(
            self._video_meta.width,
            self._video_meta.height,
        ),
    )

  def _start_server(self) -> None:
    server_cmd = [adb.ADB, '-s', str(self._device.serial)]
    server_cmd += list(_LAUNCHING_SERVER_CMD)
    if self._video_meta.bit_rate:
      server_cmd.append(f'video_bit_rate={self._video_meta.bit_rate}')
    if self._video_meta.fps:
      server_cmd.append(f'max_fps={self._video_meta.fps}')
    if self._video_meta.max_height:
      server_cmd.append(f'max_size={self._video_meta.max_height}')
    if self._video_meta.display_id is not None:
      server_cmd.append(f'display_id={self._video_meta.display_id}')
    server_cmd.append(f'capture_orientation={self._video_meta.orientation}')

    cmd_str = ' '.join(server_cmd)
    self._device.log.debug(f'Starting server with Popen command: {cmd_str}')
    self._server_proc = subprocess.Popen(
        server_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    assert self._server_proc.stdout is not None
    stdout_line = b''
    for output_line in iter(self._server_proc.stdout.readline, ''):
      if isinstance(output_line, bytes):
        stdout_line = output_line.decode('utf-8').strip()
      else:
        stdout_line = str(output_line).strip()
      if stdout_line.find('[server] INFO: Device: ') == 0:
        break
      else:
        # First line wasn't the device info line (most likely aborted)
        self._device.log.debug(f'Server stdout: {stdout_line}')
        raise Error(self._device, 'Start server failed.')
    self._device.log.debug(f'Server started: {stdout_line}.')

  def _kill_scrcpy_server(self) -> None:
    if self._server_proc:
      self._device.log.debug('Killing server process.')
      self._server_proc.kill()
      self._server_proc = None
    try:
      self._device.adb.shell(['pkill', '-f', 'scrcpy'])
    except adb.AdbError:
      pass    # Expected if no scrcpy processes are running.

  def _write_frame_to_file(self, frame_bytes: bytes) -> None:
    if self._video_writer is None:
      raise ValueError('_set_writer() not call, _video_writer is None.')
    # if frame_bytes is empty, write a blank frame.
    frame = (
        cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame_bytes
        else np.zeros(
            (self._video_meta.height, self._video_meta.width, 3), np.uint8
        )
    )
    self._video_writer.write(frame)

  def create_output_excerpts(
      self, test_info: runtime_test_info.RuntimeTestInfo) -> List[Any]:
    """Creates excerpts for the videos from the recording session.

    Args:
      test_info: The currently running test to use for determining the excerpt
        storage location.

    Returns:
      The list of location of the saved video file.
    Raises:
      Error: Raise if the video file is not found.
    """
    if self._is_alive:
      self.stop()
    self._release_video_file()
    if len(self._video_files) == 1:
      file_location = self._video_files[-1]
      target_location = os.path.join(test_info.output_path,
                                     self._output_filename)
      self._device.log.debug(f'Save video file to {target_location}')
      shutil.move(file_location, target_location)
      self._video_files = []
      self.start()
      return [target_location]
    else:
      self._device.log.info('Service writes none or more than one videos.')
      raise Error(self._device, 'Service writes none or more than one videos.')

  @retrying.retry(
      stop_max_attempt_number=_FORWARD_PORT_RETRY_TIMES,
      wait_fixed=_FORWARD_PORT_RETRY_INTERVAL_SEC,
      retry_on_exception=lambda e: isinstance(e, (adb.AdbError)),
  )
  def _forward_port(self) -> None:
    with ADB_PORT_LOCK:  # lock when allocating and forwarding port
      port = self._device.adb.forward(['tcp:0', 'localabstract:scrcpy'])
      self.port = int(port.split(b'\n')[0])
      
  def _generate_video_start_log_with_filename(
      self, device: android_device.AndroidDevice, filename: str
  ) -> str:
    """Gets the log when video record start.

    Args:
      device: The Android device which records the video.
      filename: The file name which the video record is saved as.

    Returns:
      The log contains video starting time.
    """

    return 'INFO:%s Start video recording %s, output filename %s' % (
        device.serial,
        str(
            device.adb.shell(
                'echo $(date +%Y-%m-%dT%T)${EPOCHREALTIME:10:4}'
            ).replace(b'\n', b''),
            'UTF-8',
        ),
        filename,
    )
