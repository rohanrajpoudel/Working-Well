#!/usr/bin/ python
import RPi.GPIO as GPIO
import time
import I2C_LCD_driver
mylcd = I2C_LCD_driver.lcd()
import cv2
from resultWithCamera import capture_image
import random
import datetime
# X-band radar sensing output connected to GPIO 27 of Rpi
HB100_INPUT_PIN = 17



# sample global vars for adjustment of sensitivity/measurements
#   - A greater max-pulse-count will count more pulse inputs, leading
#     to more averaged doppler frequency - it will also take longer
#   - A higher motion-sensitivity will reduce false motion readings
MAX_PULSE_COUNT = 10
MOTION_SENSITIVITY = 0


def count_frequency(GPIO_pin, max_pulse_count=10, ms_timeout=50):
    """ Monitors the desired GPIO input pin and measures the frequency
        of an incoming signal. For this example it measures the output of
        a HB100 X-Band Radar Doppler sensor, where the frequency represents
        the measured Doppler frequency due to detected motion.
    """
    start_time = time.time()
    pulse_count = 0

    # count time it takes for 10 pulses
    for _ in range(max_pulse_count):

        # wait for falling pulse edge - or timeout after 50 ms
        edge_detected = GPIO.wait_for_edge(GPIO_pin, GPIO.RISING, timeout=ms_timeout)

        # if pulse detected - iterate count
        if edge_detected is not None:
            pulse_count += 1

    # work out duration of counting and subsequent frequency (Hz)
    duration = time.time() - start_time
    frequency = pulse_count / duration
    return frequency

# loop continuously, measuring Doppler frequency and printing if motion detected
while True:
    # configure GPIO pins for input into GPIO 27
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(HB100_INPUT_PIN, GPIO.IN)
    doppler_freq = count_frequency(HB100_INPUT_PIN)
    speed = doppler_freq / float (19.49)
    freq = "{:.3f}".format(doppler_freq)
    s = "{:06.2f}".format(speed)
    mylcd.lcd_display_string("frequency:0     " ,1)
    mylcd.lcd_display_string("Speed:000.00    " ,2)
    if doppler_freq > MOTION_SENSITIVITY:
       mylcd.lcd_display_string("frequency:%s" %freq,1)
       mylcd.lcd_display_string("Speed:%s" %s,2)
    if speed > 1:
        originalImage, finalResultImage = capture_image()
        cv2.imwrite("../SavedItem/O"+str(datetime.datetime.now())+".jpg", originalImage)
        cv2.imwrite("../SavedItem/F"+str(datetime.datetime.now())+".jpg", finalResultImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    GPIO.cleanup()