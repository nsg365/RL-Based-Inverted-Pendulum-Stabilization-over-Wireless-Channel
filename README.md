# RL Based Inverted Pendulum over Wireless Channel 
### Wired + Wireless Implementation

---

##  Overview

This project implements control of an inverted pendulum system using both **wired** (Arduino-based) and **wireless** (ESP-NOW-based) architectures.

The system is modeled and analyzed in **MATLAB/Simulink**, and control is implemented using a **proportional controller** derived via pole placement techniques.

---

##  System Architecture

### 1. Wired Control (Arduino)
- Sensor readings are directly obtained by the Arduino
- Control computation is performed locally
- Output voltage is applied directly to the motor

### 2. Wireless Control (ESP32 + Arduino)
- Sensor data is collected by an ESP32 (plant side)
- Data is transmitted via **ESP-NOW**
- Controller ESP32 computes control input
- Control signal is sent back to plant ESP
- Arduino Mega acts as a voltage interface (**3.3V → 5V**) for motor actuation

---

##  Repository Structure

```
proportional/
    Proportional_controller.ino

simulation/
    controlled_simulation.m
    linearization_with_motor.m
    pendulum_model.slx
    pole_placement_with_motor.m

wireless/
    Delay_RTT.ino
    ESP_PWM_reader.ino
    Pendulum_controller.ino
    Pendulum_plant.ino
```

---

##  Simulation (MATLAB / Simulink)

### Files

| File | Description |
|------|-------------|
| `linearization_with_motor.m` | Linearizes the pendulum system including motor dynamics |
| `pole_placement_with_motor.m` | Computes controller gains using pole placement |
| `controlled_simulation.m` | Simulates closed-loop system performance |
| `pendulum_model.slx` | Simscape model of pendulum + motor dynamics |

### Purpose
- Model system dynamics
- Include motor voltage behavior
- Design controller gains before hardware implementation

---

##  Proportional Control (Wired Implementation)

**File:** `proportional/Proportional_controller.ino`

- Implements proportional controller on Arduino
- Uses gains obtained from simulation
- Sensor readings are directly processed
- Motor is controlled via wired interface

---

##  Wireless Control (ESP-NOW Implementation)

### `Pendulum_plant.ino`
- Runs on ESP32 (plant side)
- Reads sensor data
- Transmits data to controller via ESP-NOW
- Receives control signal

### `Pendulum_controller.ino`
- Runs on ESP32 (controller side)
- Receives sensor data
- Computes control input (voltage)
- Sends control signal back to plant

### `ESP_PWM_reader.ino`
- Runs on Arduino Mega
- Reads PWM signal from ESP32 (3.3V)
- Converts it to 5V signal for motor driver
- Acts as a level shifter + actuator interface

### `Delay_RTT.ino`
- Measures communication delay (Round Trip Time)
- Evaluates latency in wireless control loop

---

##  Key Features

- Dual architecture: wired and wireless control
- Real-time ESP-NOW communication
- Latency measurement and analysis
- MATLAB-based system modeling and controller design
- Integration of embedded systems with control theory

---

##  Applications

- Networked control systems
- Wireless feedback control
- Real-time embedded control systems
- Robotics and automation

---

##  Notes

- Controller gains are derived from MATLAB simulations
- Wireless system introduces delay, analyzed via RTT measurement
- Arduino Mega is used to handle voltage level mismatch (3.3V → 5V)
