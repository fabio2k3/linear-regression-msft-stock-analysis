import os

model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
print("Ruta models:", os.path.abspath(model_dir))
print("\nArchivos dentro de models:")
for f in os.listdir(model_dir):
    print("->", repr(f))
