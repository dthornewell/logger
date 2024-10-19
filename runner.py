from time import sleep
from uart import UART_Driver

driver = UART_Driver("COM6")
driver.reset_kart()
for i in range(21):
    driver.update_velocity(40 + i)
    driver.update_steering(i * 2)
    driver.write_serial()
    print(f"steering at {i}")
    sleep(0.05)
sleep(1.5)
for i in range(21):
    driver.update_velocity(60 - i)
    driver.update_steering(40 - i * 2)
    driver.write_serial()
    print(f"steering at {i}")
    sleep(0.05)
driver.reset_kart()
