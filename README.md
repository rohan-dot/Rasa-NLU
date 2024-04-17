# Rasa-NLU
Interning for convergys I made a chatbot for password reset using rasa-nlu.
I trained the bot for password reset functionalities adding the T-opt service and Lanid of the employee and trained it via own examples to make it's answer seem more human like rather than robotic



 from PIL import Image

# Function to overlay two images
def overlay_images(background_path, overlay_path, output_path, position=(0, 0)):
    # Open the background and overlay images
    background = Image.open(background_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA")

    # Create a new image with the same size as the background, transparent initially
    composite = Image.new("RGBA", background.size)

    # Paste the background image to the composite image
    composite.paste(background, (0, 0))

    # Paste the overlay image on top of the background image at the given position
    composite.paste(overlay, position, overlay)  # overlay used as a mask for itself

    # Save the final image
    final_image = composite.convert("RGBA")  # Ensure the final image is in RGBA mode
    final_image.save(output_path)

# Paths to your images
background_image_path = 'path_to_background_image.png'
overlay_image_path = 'path_to_overlay_image.png'
output_image_path = 'path_to_output_image.png'

# Call the function with the path to your images and the desired position
overlay_images(background_image_path, overlay_image_path, output_image_path, position=(50, 50))
