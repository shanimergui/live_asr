import tkinter as tk

# Create the main window
window = tk.Tk()

# Define a function to be called when the button is clicked
def some_function():
    print("Button was clicked!")

# Create the button
button = tk.Button(text="Click me!", command=some_function)

# Pack the button to show it
button.pack()

# Run the Tkinter event loop
window.mainloop()
