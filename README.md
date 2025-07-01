# SENTIMENT-PREDICTION

Sentiment and Intent Classification with Ollama API
This Python script classifies free-text messages into predefined sentiment and intent categories using the Ollama language model API. It processes a CSV file containing messages and outputs a CSV file with the original messages and their classified sentiment categories.

Features
Uses the Ollama API to classify each message into one of several sentiment or intent categories.

Custom prompt designed with specific classification rules for different types of inputs (numeric ratings, inquiries, complaints, informal words, etc.).

Supports batch processing of messages stored in a CSV file.

Outputs results to a new CSV file with sentiment categories alongside the original messages.

Provides progress feedback during processing with tqdm.

Requirements
Python 3.7+

pandas

requests

tqdm

You can install the required packages using pip:

bash
Copy
Edit
pip install pandas requests tqdm
Usage
Prepare your input CSV file:

The script expects a CSV file named FreeTextData1.csv with at least two columns:

freetextid: unique identifier for each message

messages: the free-text message to classify

Run the script:

bash
Copy
Edit
python classify_messages.py
Output:

The script creates a CSV file named messages_with_sentimenit_ollama.csv containing:

freetextid

messages

sentiment_category (the predicted category by the model)

Configuration
OLLAMA_MODEL - Set this to the name of the Ollama model you want to use (default "llama3").

OLLAMA_URL - The API endpoint for your local Ollama server (default "http://localhost:11434/api/generate").

How It Works
The script reads each message from the CSV.

It constructs a detailed prompt specifying the classification categories and rules.

The prompt is sent to the Ollama model API for classification.

The response (category) is parsed and saved alongside the original message.

Messages with freetextid > 1092866 are processed (can be adjusted as needed).

Classification Categories
Excellent

Very Good

Good

Average

Poor

Inquiry

Car Not Recieved

Vehicle Sold

Mobile Number Wrong

Bill Issue

Others

Detailed rules and priority are encoded in the prompt for consistent classification.

Notes
Make sure your Ollama API server is running locally and accessible at the specified URL.

The script handles errors gracefully by assigning the "Others" category in case of API failures.

Adjust the filtering condition (if msg_id > 1092866:) as needed to control which messages to classify.

