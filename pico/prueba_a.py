# Prueba A: amplificador (U1/U2) + receptor + emisor HC-SR04 #3
from machine import Pin, ADC
import time

trig = Pin(6, Pin.OUT, value=0)   # emisor HC-SR04 #3
comp = Pin(7, Pin.IN)             # comparador (U2 pin 7)
adc  = ADC(26)                    # senal amplificada (U2 pin 1)

def disparo():
    trig.value(1); time.sleep_us(10); trig.value(0)

while True:
    # 1) amplitud de la senal (pico a pico)
    disparo()
    lo, hi = 65535, 0
    for _ in range(1500):
        v = adc.read_u16()
        lo = min(lo, v); hi = max(hi, v)
    pp = (hi - lo) * 3.3 / 65535
    reposo = adc.read_u16() * 3.3 / 65535

    # 2) tiempo del eco con el comparador
    time.sleep_ms(30)
    disparo()
    t0 = time.ticks_us()
    while time.ticks_diff(time.ticks_us(), t0) < 800:   # ignora el acople directo
        pass
    t = None
    while time.ticks_diff(time.ticks_us(), t0) < 25000:
        if comp.value():
            t = time.ticks_diff(time.ticks_us(), t0); break

    dist = round(t * 0.0343 / 2, 1) if t else None
    print("reposo: %.2f V | pico-pico: %.2f V | eco: %s us | ~%s cm" % (reposo, pp, t, dist))
    time.sleep_ms(200)
