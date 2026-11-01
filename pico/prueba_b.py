# Prueba B: HC-SR04 #1 y #2 + servos PAN/TILT
from machine import Pin, PWM, time_pulse_us
import time

def medir(trig, echo):
    trig.value(0); time.sleep_us(2)
    trig.value(1); time.sleep_us(10); trig.value(0)
    t = time_pulse_us(echo, 1, 30000)
    return round(t * 0.0343 / 2, 1) if t > 0 else None

t1, e1 = Pin(2, Pin.OUT), Pin(3, Pin.IN)   # HC-SR04 #1
t2, e2 = Pin(4, Pin.OUT), Pin(5, Pin.IN)   # HC-SR04 #2
pan, tilt = PWM(Pin(15)), PWM(Pin(14))
for s in (pan, tilt):
    s.freq(50)

def angulo(servo, a):                      # 0-180 grados -> pulso 0.5-2.5 ms
    servo.duty_ns(int(500_000 + a * 2_000_000 / 180))

angulo(tilt, 90)
for a in (0, 45, 90, 135, 180, 90):
    angulo(pan, a)
    time.sleep(0.7)
    print("PAN", a, "| #1:", medir(t1, e1), "cm | #2:", medir(t2, e2), "cm")
