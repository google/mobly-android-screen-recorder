# Mobly Android Screen Recorder

Mobly Android Screen Recorder service for using Python code to screencast the
Android devices in Mobly tests.

## Requirements

-   Python 3.11+
-   Mobly 1.12.2+

## Installation

```shell
pip install mobly-android-screen-recorder
```

## Start to Use

After initializing the Android device, you can register the screen recorder
service with the following code:

```python
from mobly.controllers.android_device_lib.services.android_screen_recorder import screen_recorder
...

self.dut = self.register_controller(android_device)[0]
self.dut.services.register('screen_recorder', screen_recorder.ScreenRecorder)
```

Then the screen recorder will start recording the screen when the test starts
and stop recording when the test finishes. The screen recording will be saved to
the test output folder as a video file during teardown process for each test
case with **create_output_excerpts_all** for all registered Mobly services.

## Example 1: Hello World!

  Let's start with the simple example of posting "Hello World" on the Android
device's screen. Create the following files:   **sample_config.yml**

```yaml
TestBeds:
  # A test bed where adb will find Android devices.
  - Name: SampleTestBed
    Controllers:
        AndroidDevice: '*'
```

**hello_world_test.py**

```python
from mobly import base_test
from mobly import test_runner
from mobly.controllers import android_device

class HelloWorldTest(base_test.BaseTestClass):

  def setup_class(self):
    # Registering android_device controller module declares the test's
    # dependency on Android device hardware. By default, we expect at least one
    # object is created from this.
    self.ads = self.register_controller(android_device)
    self.dut = self.ads[0]
    # Start Mobly Bundled Snippets (MBS).
    self.dut.load_snippet('mbs', android_device.MBS_PACKAGE)
    # Register screen recorder service, it will start recording when the test
    # starts and stop recording when the test finishes.
    self.dut.services.register('screen_recorder', screen_recorder.ScreenRecorder)

  def test_hello(self):
    self.dut.mbs.makeToast('Hello World!')

  def teardown_test(self):
    self.dut.services.create_output_excerpts_all(self.current_test_info)

if __name__ == '__main__':
  test_runner.main()
```

To execute:

```
$ python hello_world_test.py -c sample_config.yml
```

*Expect*:

A "Hello World!" toast notification appears on your device's screen. And a video
file named `video,{device_serial},{device_model},{timestamp}.mp4` is created in
the test output folder.
