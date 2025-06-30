# Scribble Game: AI-Powered Drawing Guessing Game

This project is a web-based interactive drawing game where users draw a word on a canvas, and an Artificial Intelligence model attempts to guess what is being drawn in real-time. It's built using Django for the backend and HTML/CSS/JavaScript for the frontend, demonstrating a machine learning model integrated into a real-time web application.

## Features

* **Interactive Drawing Canvas:** Users can draw freely on a dedicated canvas.
* **Real-time AI Prediction:** An integrated AI model continuously analyzes the drawing and provides its guesses in real-time.
* **Word Challenges:** Users select from a set of pre-defined words to draw, provided at the start of each game.
* **Hint System:** The AI model receives hints (revealed letters of the target word) if it struggles to guess, aiding in its prediction.
* **AI Chatbox:** Model predictions and confidence levels are displayed in a chat-like interface for user feedback.
* **Game Timer:** Each drawing challenge has a time limit.

## Repository Structure

The project follows a standard Django application structure:

Scribble_Project/

├── drawapp/                     # The main Django application

│   ├── migrations/

│   ├── templates/

│   │   └── drawapp/

│   │       ├── draw_page.html   # Frontend for the drawing game

│   │       └── username_page.html # User login/username page

│   ├── init.py

│   ├── admin.py

│   ├── apps.py

│   ├── classes.txt              # List of words the AI model can recognize

│   ├── models.py

│   ├── step_176000.keras        # Pre-trained AI model (Keras format)

│   ├── tests.py

│   ├── urls.py                  # URL configurations for drawapp

│   └── views.py                 # Backend logic for handling requests

├── scribble_project/            # The main Django project configuration

│   ├── init.py

│   ├── asgi.py

│   ├── settings.py              # Project settings

│   ├── urls.py                  # Main project URL configurations

│   └── wsgi.py

├── db.sqlite3                   # Default SQLite database file

├── manage.py                    # Django's command-line utility

└── requirements.txt             # Project dependencies 

## Core Components Explained

### `draw_page.html`

This HTML file represents the client-side interface of the drawing game.

* **User Interface:** It provides the visual elements for the game, including:
    * A `<canvas>` element where users can draw using their mouse.
    * Buttons to clear the canvas and start a new challenge.
    * A list of challenge words for the user to choose from.
    * Displays for the selected word, hint status, and a countdown timer.
    * A chatbox (`#chatbox`) to display the AI's real-time guesses.
* **Client-Side Logic (JavaScript):**
    * Handles drawing mechanics (tracking mouse movements to draw lines on the canvas).
    * Manages the game flow, including word selection and timer updates.
    * Communicates with the Django backend by sending drawing image data (as base64 strings) to the `/predict/` endpoint using `fetch` API calls.
    * Receives predictions from the backend and updates the chatbox with the AI's guesses and confidence levels.
    * Contains logic for revealing hints to the model at certain time intervals if the model hasn't guessed correctly.

### `views.py`

This Python file contains the backend logic for the Django application, processing requests and interacting with the AI model.

* **Model Loading:**
    * It loads a pre-trained Keras deep learning model (`step_176000.keras`) and the list of recognizable classes/words from `classes.txt` once when the server starts.
* **Page Rendering:**
    * `username_page(request)`: Handles the initial username input, setting it in the session before redirecting to the drawing page.
    * `draw_page(request)`: Renders the `draw_page.html` template, dynamically selecting and passing three random challenge words from `classes.txt` to the frontend.
* **Prediction Endpoint (`predict(request)`):**
    * This is the core AI inference endpoint, accessed via a POST request from the frontend.
    * It receives the base64 encoded image of the current drawing, the user's selected word, and any revealed hints.
    * **Image Preprocessing:** It employs a function `extract_and_resize_parts` to create multiple augmented versions of the input image (original resized, overlapping parts, center crop, and an optional 90-degree rotation). This helps the model generalize better and improve prediction accuracy.
    * **AI Inference:** It runs predictions on all the processed image parts using the loaded Keras model.
    * **Prediction Filtering:** It intelligently filters the raw model predictions based on several criteria:
        * **Length Matching:** Prioritizes words that match the length of the actual target word.
        * **Hint Application:** Filters predictions based on any revealed hints (e.g., if the hint is 'a' at index 0, only words starting with 'a' are considered).
        * **Randomness:** Introduces a small chance to skip some filtering steps, mimicking less predictable AI behavior.
        * **Previous Failed Guesses:** Keeps track of previous incorrect guesses and avoids repeating them (unless it's the correct answer).
    * **Response Generation:** Returns a JSON response containing the top predicted words with their confidence scores, and also `visual_inputs` which include base64 images of what the model "saw" (the preprocessed parts) and their individual top predictions.
    * **Simulated Delay:** Includes a slight delay to simulate processing time and provide a more natural user experience.

## How to Run

To set up and run this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/sp2772/Scribble_Project.git](https://github.com/sp2772/Scribble_Project.git)
    cd Scribble_Project
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    This project requires Django, TensorFlow (for Keras), OpenCV, Pillow, and NumPy. While a `requirements.txt` is not provided in the prompt, you would typically install them as follows:
    ```bash
    pip install django tensorflow opencv-python pillow numpy
    ```
    *(Note: The exact TensorFlow version compatible with the `.keras` model might be specific. If you encounter issues, try TensorFlow 2.x versions.)*

4.  **Run Database Migrations:**
    Navigate to the outer `scribble_project` directory (where `manage.py` is located) and apply migrations.
    ```bash
    cd scribble_project
    python manage.py migrate
    ```

5.  **Start the Development Server:**
    From the same `scribble_project` directory:
    ```bash
    python manage.py runserver
    ```

6.  **Access the Application:**
    Open your web browser and navigate to:
    `http://127.0.0.1:8000/username/`

    You will first be prompted to enter a username, and then redirected to the drawing game.
