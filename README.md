# Mobly Android Screen Recorder

Mobly Android Screen Recorder service for using Python code to screencast the
Android devices in Mobly tests (https://github.com/google/mobly).

## Responsible Use

This tool is designed for legitimate testing and debugging purposes within the
context of Mobly. It should only be used in situations where you have:

* **Explicit consent:** Obtain clear and informed consent from any individuals
* whose screens might be recorded.
* **Legitimate purpose:**  Use the tool for legitimate testing, debugging,
* or development activities related to Mobly.
* **Data security:**  Handle recorded screen data responsibly. Store it
* securely and delete it when it's no longer needed.
* **Transparency:** Be transparent about how the tool is being used and what
* data is being collected.

**Misuse of this tool for unauthorized surveillance or any other malicious
activity is strictly prohibited.**

**Remember:** Recording someone's screen without their knowledge or consent
is a serious breach of privacy and may have legal consequences. 

## Requirements

-   Python 3.11+
-   Mobly 1.12.2+
-   FFMPEG 6.1.1+

Please make sure your FFMPEG support H.264 (run `ffmpeg -codecs` and
look for **libx264** encoder in the output) 

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

## Disclaimer

This tool, owned by Google and its developers, provides screen recording functionality for Android devices within the context of Mobly testing. It is essential to use this tool responsibly and ethically. 

**Privacy Warning:** Screen recording can capture sensitive information. **Always obtain explicit consent before recording anyone's screen.** Unauthorized screen recording may violate privacy laws and ethical guidelines. Neither Google nor the developers of this tool are responsible for any misuse or illegal activity conducted with this software.

Users of this tool are solely responsible for ensuring compliance with all applicable laws and regulations regarding privacy and data protection.

## Licensing

This project is licensed under the [Apache License 2.0](LICENSE).

This project uses [scrcpy](https://github.com/Genymobile/scrcpy), which is licensed under the Apache License 2.0. See the [NOTICE](NOTICE) file for details.
