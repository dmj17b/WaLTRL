import pygame

# Initialize essential Pygame systems
pygame.init()
pygame.display.set_mode((1, 1)) # Needed to keep the event queue active on some OSs
pygame.joystick.init()

# Check for connected controllers
if pygame.joystick.get_count() == 0:
    print("Please connect a joystick or gamepad.")
else:
    # Initialize the first connected joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Reading axes for: {joystick.get_name()}")

# Main event loop
while True:
    pygame.event.pump()  # Process event queue to update joystick state
    for var in range(joystick.get_numaxes()):
        axis_value = joystick.get_axis(var)
        print(f"Axis {var}: {axis_value:.2f}")
    for button in range(joystick.get_numbuttons()):
        button_value = joystick.get_button(button)
        print(f"Button {button}: {button_value}")
    pygame.time.wait(100)  # Wait a bit before the next read to avoid spamming the console
pygame.quit()