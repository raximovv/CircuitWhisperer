"""
CircuitWhisperer — Day 22

Requirements:
    pip install opencv-python pillow numpy

Also install/run:
    ollama serve
    ollama pull moondream

Run:
    python day22_circuitwhisperer.py
"""

import os
import sys
import time
import json
import tempfile
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw


MODEL = "moondream"
MAX_SIZE = 512


COMPONENT_PROMPT = """
You are analyzing a hand-drawn electronic circuit schematic.

Return a structured component list.

Rules:
- Only list components that are visible.
- Do not invent components.
- Include confidence for each item.
- If unsure, say "unknown symbol".
- Keep output concise.

Format exactly like this:

Components:
1. name: resistor, label: R1, confidence: high
2. name: capacitor, label: C1, confidence: medium
3. name: ground, label: GND, confidence: high
"""

FUNCTION_PROMPT = """
You are analyzing an electronic circuit schematic.

Explain what the circuit likely does.

Rules:
- Be concise.
- Mention signal flow if visible.
- Mention uncertainty if the drawing is unclear.
- Do not hallucinate hidden components.

Answer in 2-4 sentences.
"""

WIRING_ERROR_PROMPT = """
Look carefully at this circuit schematic.

Check for possible wiring or schematic problems:
- missing ground
- floating input or output
- disconnected node
- short circuit
- reversed polarity
- unclear connection
- impossible component placement

Return up to 3 possible issues.

Format:
Wiring check:
1. issue: ..., confidence: high/medium/low
2. issue: ..., confidence: high/medium/low

If no obvious issue is visible, say:
No obvious wiring errors detected.
"""

FOLLOW_UP_PROMPT_TEMPLATE = """
You are answering a follow-up question about this circuit schematic.

Question:
{question}

Rules:
- Answer based only on the visible circuit.
- Be honest if the image is unclear.
- Keep the answer short and useful.
"""


def check_ollama_setup():
    """Check that Ollama and the model are available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=8,
        )

        if result.returncode != 0:
            print("ERROR: Ollama is not running.")
            print("Run this in another terminal:")
            print("  ollama serve")
            sys.exit(1)

        if MODEL.lower() not in result.stdout.lower():
            print(f"ERROR: Model '{MODEL}' not found.")
            print("Install it with:")
            print(f"  ollama pull {MODEL}")
            sys.exit(1)

        print(f"✓ Ollama ready with model: {MODEL}")

    except FileNotFoundError:
        print("ERROR: Ollama is not installed or not in PATH.")
        print("Install Ollama first, then run:")
        print(f"  ollama pull {MODEL}")
        sys.exit(1)

    except subprocess.TimeoutExpired:
        print("ERROR: Ollama check timed out.")
        print("Make sure Ollama is running:")
        print("  ollama serve")
        sys.exit(1)


def generate_test_schematic(output_path="test_circuit.png"):
    """Generate a simple RC low-pass filter schematic."""
    width, height = 700, 420
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    black = "black"
    line_width = 4

    draw.text((250, 25), "RC Low-Pass Filter", fill=black)

    # VIN
    draw.text((35, 185), "VIN", fill=black)
    draw.line((85, 195, 140, 195), fill=black, width=line_width)

    # Resistor zig-zag
    draw.text((190, 145), "R1 10k", fill=black)

    x, y = 140, 195
    points = [(x, y)]
    step = 24

    for i in range(8):
        px = x + step * (i + 1)
        py = y - 18 if i % 2 == 0 else y + 18
        points.append((px, py))

    points.append((360, y))
    draw.line(points, fill=black, width=line_width)

    # Output node
    draw.line((360, 195, 470, 195), fill=black, width=line_width)
    draw.ellipse((355, 190, 365, 200), fill=black)
    draw.text((480, 185), "VOUT", fill=black)

    # Capacitor to ground
    cap_x = 395
    draw.text((415, 230), "C1 100nF", fill=black)

    draw.line((cap_x, 195, cap_x, 245), fill=black, width=line_width)
    draw.line((cap_x - 35, 245, cap_x + 35, 245), fill=black, width=line_width)
    draw.line((cap_x - 35, 265, cap_x + 35, 265), fill=black, width=line_width)
    draw.line((cap_x, 265, cap_x, 320), fill=black, width=line_width)

    # Ground symbol
    draw.line((cap_x - 35, 320, cap_x + 35, 320), fill=black, width=line_width)
    draw.line((cap_x - 22, 335, cap_x + 22, 335), fill=black, width=line_width)
    draw.line((cap_x - 10, 350, cap_x + 10, 350), fill=black, width=line_width)
    draw.text((cap_x + 45, 315), "GND", fill=black)

    img.save(output_path)
    print(f"✓ Generated test schematic: {output_path}")
    return output_path


def preprocess_image(image_path, max_size=MAX_SIZE):
    """Resize and improve contrast for vision model input."""
    img = cv2.imread(image_path)

    if img is None:
        return None

    height, width = img.shape[:2]
    scale = max_size / max(height, width)

    if scale < 1:
        img = cv2.resize(
            img,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast for pencil/pen drawings
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        8,
    )

    processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp.name, processed)

    return temp.name


def query_ollama_vision(image_path, prompt):
    """Send an image and prompt to Ollama Moondream."""
    try:
        command = [
            "ollama",
            "run",
            MODEL,
            f"{prompt}\n\nImage: {image_path}",
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=90,
        )

        if result.returncode != 0:
            return f"[Ollama error]\n{result.stderr.strip()}"

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return "[Model timed out. Try a smaller or clearer image.]"

    except Exception as error:
        return f"[Error: {error}]"


def analyze_circuit(image_path):
    """Run component, function, and wiring-error analysis."""
    processed_path = preprocess_image(image_path)

    if processed_path is None:
        print("ERROR: Could not load image.")
        return None

    print("\n" + "=" * 60)
    print("CircuitWhisperer Analysis")
    print("=" * 60)

    print("\n[1/3] Detecting components...")
    start = time.time()
    components = query_ollama_vision(processed_path, COMPONENT_PROMPT)
    print(f"Done in {time.time() - start:.1f}s")
    print("\n" + components)

    print("\n[2/3] Explaining circuit function...")
    start = time.time()
    function = query_ollama_vision(processed_path, FUNCTION_PROMPT)
    print(f"Done in {time.time() - start:.1f}s")
    print("\n" + function)

    print("\n[3/3] Checking wiring errors...")
    start = time.time()
    wiring = query_ollama_vision(processed_path, WIRING_ERROR_PROMPT)
    print(f"Done in {time.time() - start:.1f}s")
    print("\n" + wiring)

    print("\n" + "=" * 60)

    result = {
        "image": image_path,
        "components": components,
        "function": function,
        "wiring_errors": wiring,
        "processed_image": processed_path,
    }

    return result


def follow_up_loop(processed_image_path):
    """Allow user to ask follow-up questions about the circuit."""
    print("\nAsk follow-up questions about the circuit.")
    print("Examples:")
    print("  What type of filter is this?")
    print("  What does R1 do?")
    print("  Is the capacitor connected correctly?")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Follow-up question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        if not question:
            continue

        prompt = FOLLOW_UP_PROMPT_TEMPLATE.format(question=question)

        print("\nThinking...")
        answer = query_ollama_vision(processed_image_path, prompt)
        print("\nAnswer:")
        print(answer)
        print()


def load_image_from_file():
    """Find a circuit image in the current folder."""
    possible_files = [
        "circuit.jpg",
        "circuit.jpeg",
        "circuit.png",
        "schematic.jpg",
        "schematic.png",
        "test_circuit.png",
    ]

    for filename in possible_files:
        if os.path.exists(filename):
            return filename

    return None


def capture_from_webcam():
    """Capture a circuit photo from webcam."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("ERROR: No webcam found.")
        return None

    print("\nWebcam opened.")
    print("Press SPACE to capture.")
    print("Press Q to quit webcam.")

    captured_path = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        display = frame.copy()

        cv2.putText(
            display,
            "SPACE = capture | Q = quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("CircuitWhisperer Webcam", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(temp.name, frame)
            captured_path = temp.name
            print(f"✓ Captured image: {captured_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_path


def save_results(result, output_path="analysis_result.txt"):
    """Save analysis output for README/screenshots."""
    if result is None:
        return

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("CircuitWhisperer Analysis Result\n")
        file.write("=" * 40 + "\n\n")

        file.write("Image:\n")
        file.write(result["image"] + "\n\n")

        file.write("Components:\n")
        file.write(result["components"] + "\n\n")

        file.write("Circuit Function:\n")
        file.write(result["function"] + "\n\n")

        file.write("Wiring Errors:\n")
        file.write(result["wiring_errors"] + "\n")

    print(f"\n✓ Results saved to {output_path}")


def main():
    check_ollama_setup()

    print("\n" + "=" * 60)
    print("CircuitWhisperer — Day 22")
    print("=" * 60)

    print("\nChoose input:")
    print("1. Use generated test schematic")
    print("2. Load circuit image from current folder")
    print("3. Capture from webcam")
    print("4. Enter custom image path")
    print("q. Quit")

    choice = input("\nSelect option: ").strip().lower()

    image_path = None
    captured_temp = False

    if choice == "1":
        image_path = generate_test_schematic()

    elif choice == "2":
        image_path = load_image_from_file()

        if image_path is None:
            print("No image found.")
            print("Put an image named circuit.jpg, circuit.png, schematic.jpg, or schematic.png in this folder.")
            return

        print(f"✓ Loaded image: {image_path}")

    elif choice == "3":
        image_path = capture_from_webcam()
        captured_temp = True

        if image_path is None:
            return

    elif choice == "4":
        custom_path = input("Enter image path: ").strip()

        if not os.path.exists(custom_path):
            print("ERROR: File does not exist.")
            return

        image_path = custom_path

    elif choice == "q":
        print("Goodbye.")
        return

    else:
        print("Invalid choice.")
        return

    result = analyze_circuit(image_path)

    if result is not None:
        save_results(result)

        answer = input("\nDo you want to ask follow-up questions? (y/n): ").strip().lower()

        if answer == "y":
            follow_up_loop(result["processed_image"])

        try:
            os.unlink(result["processed_image"])
        except Exception:
            pass

    if captured_temp and image_path:
        try:
            os.unlink(image_path)
        except Exception:
            pass

    print("\nDone. Ship it 🐋")


if __name__ == "__main__":
    main()