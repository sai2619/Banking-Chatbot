# Banking Chatbot

## Overview

This repository contains code for a Banking Chatbot implemented using Python and Streamlit. The chatbot allows users to interact with it through text or voice input, assisting them with various banking tasks such as checking account balances, transferring money, and querying dues.

## Dependencies

To run the Banking Chatbot, ensure you have the following dependencies installed:

1. Python 3.x
2. Streamlit
3. SoundDevice
4. NumPy
5. Torch
6. Vosk
7. Pyttsx3
8. Wave
9. Langchain
10. Transformers
11. OpenAI

You can install these dependencies using pip with the following command:

```
pip install -r requirements.txt
```
We can import all the above dependencies in the requirements.txt file.

## Usage

1. Install the required dependencies using the command mentioned above.
2. Run the Streamlit application using the following command:

```
streamlit run app.py
```

## Configuration

Before running the chatbot, make sure to configure the following:

1. Modify the JSON file path (`json_file_path`) in the code to point to the location of the account details JSON file.
2. Set up the OpenAI API key in the environment variable `OPEN_API_KEY` for accessing the language model.

## Notes

- Ensure that you have the necessary permissions and access rights for the account details JSON file.
- The OpenAI API key is required for language model interactions. Make sure to obtain the API key and set it up correctly.

## Contributing

If you find any issues or have suggestions for improvements, feel free to contribute by creating a pull request or opening an issue in the repository.