import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
from tensorflow.keras import models

# ========================
# Part 2: Tkinter GUI for Drawing & Prediction
# ========================

# Load the pre-trained model (ensure the file name matches the one you saved)
MODEL_PATH = "improved_mnist_cnn.h5"
model = models.load_model(MODEL_PATH)
print("[INFO] Loaded model from disk.")

def predict_digit(image):
    # 1) Ensure it's 28x28
    image = image.resize((28, 28))
    # 2) Convert to grayscale
    image = image.convert('L')
    # 3) Invert to match MNIST (digit = white, background = black)
    image = ImageOps.invert(image)
    # 4) Convert to numpy
    image = np.array(image)
    # 5) Reshape for the model and normalize
    image = image.reshape(1, 28, 28, 1) / 255.0
    # 6) Predict
    pred = model.predict(image)
    return np.argmax(pred), np.max(pred)

class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")

        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        self.label = Label(root, text="Draw a digit and click 'Predict'", font=("Helvetica", 16))
        self.label.pack()

        self.button_predict = Button(root, text="Predict", command=self.predict)
        self.button_predict.pack(side=tk.LEFT)

        self.button_clear = Button(root, text="Clear", command=self.clear)
        self.button_clear.pack(side=tk.RIGHT)

        # Create a PIL image to store drawings
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse dragging event to the paint function
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Brush size can be adjusted if needed
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        # Draw on the canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        # Draw on the PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill="black", outline="black")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a digit and click 'Predict'")

    def predict(self):
        bbox = self.image.getbbox()
        if bbox:
            cropped_image = self.image.crop(bbox)
            # Directly resize to 20x20 to ensure the digit isn't too small
            cropped_image = cropped_image.resize((20, 20), Image.Resampling.LANCZOS)

            # Create a new 28x28 image with white background
            new_image = Image.new("L", (28, 28), "white")
            # Center the 20x20 digit
            new_image.paste(
                cropped_image,
                ((28 - cropped_image.width) // 2, (28 - cropped_image.height) // 2)
            )

            # Optional: Add a small Gaussian blur to smooth lines
            new_image = new_image.filter(ImageFilter.GaussianBlur(radius=1))

            digit, accuracy = predict_digit(new_image)
            self.label.config(text=f"Predicted Digit: {digit}, Accuracy: {accuracy:.2f}")
        else:
            self.label.config(text="Please draw a digit first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
