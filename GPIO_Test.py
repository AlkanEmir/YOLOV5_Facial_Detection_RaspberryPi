import RPi.GPIO as GPIO
import time

LED_INPUT = 18
GPIO.setmode(GPIO.BCM)

GPIO.setup(LED_INPUT, GPIO.OUT)
GPIO.output(LED_INPUT, GPIO.HIGH)
time.sleep(10)
GPIO.output(LED_INPUT, GPIO.LOW)
GPIO.cleanup()
