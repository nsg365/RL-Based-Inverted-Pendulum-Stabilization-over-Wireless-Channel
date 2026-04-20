// --- PIN DEFINITIONS ---
const int ESP_PWM_IN = 20; // Port D, Bit 1 (PD1) - Interrupt capable
const int ESP_DIR_IN = 21; // Port D, Bit 0 (PD0) - Interrupt capable

const int SHIELD_PWM_OUT = 5; // Port E, Bit 3 (PE3)
const int SHIELD_DIR_OUT = 4; // Port G, Bit 5 (PG5)

void setup() {
  // Standard initialization is fine here as it only runs once
  pinMode(ESP_PWM_IN, INPUT);
  pinMode(ESP_DIR_IN, INPUT);
  pinMode(SHIELD_PWM_OUT, OUTPUT);
  pinMode(SHIELD_DIR_OUT, OUTPUT);

  // Attach the hardware interrupt to Pin 20
  attachInterrupt(digitalPinToInterrupt(ESP_PWM_IN), mirrorPWM, CHANGE);
}

void loop() {
  // Mirroring Direction: PIND reads Port D. PORTG writes to Port G.
  
  if (PIND & (1 << 0)) {     // If Pin 21 (Bit 0 of Port D) is HIGH
    PORTG |= (1 << 5);       // Set Pin 4 (Bit 5 of Port G) HIGH
  } else {
    PORTG &= ~(1 << 5);      // Set Pin 4 LOW
  }
}


void mirrorPWM() {
  // Mirroring PWM: PIND reads Port D. PORTE writes to Port E.
  
  if (PIND & (1 << 1)) {     // If Pin 20 (Bit 1 of Port D) is HIGH
    PORTE |= (1 << 3);       // Set Pin 5 (Bit 3 of Port E) HIGH
  } else {
    PORTE &= ~(1 << 3);      // Set Pin 5 LOW
  }
}