import sounddevice as sd
import numpy as np
import torch
import pyttsx3
import wave
import re
import uuid
import vosk
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os
import json
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

# Upload the Vosk model
vosk_model = vosk.Model(r"C:\Users\SAI PRABHATH\Downloads\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15")

#Take OPEN_API_KEY 
os.environ["OPEN_API_KEY"]="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"
chat_model = ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"], temperature=0.5)

# JSON file which contains our account details
json_file_path = r"C:\Users\SAI PRABHATH\Downloads\database.json"

model = BertForSequenceClassification.from_pretrained(r"C:\Users\SAI PRABHATH\Downloads\Model_bert_ds_b1\Model_bert_ds_b1")

tokenizer = BertTokenizer.from_pretrained(r"C:\Users\SAI PRABHATH\Downloads\Model_bert_ds_b1\Model_bert_ds_b1")

# Recording the audio
def record_audio(voicefile="audio.wav", duration=5, sample_rate=16000):
    st.info("Recording...")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    st.info("Recorded.")
    with wave.open(voicefile, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return voicefile

# Recognizing the speech from the audio file
def recognize_speech(wav_voicefile):
    wf = wave.open(wav_voicefile, 'rb')
    sample_rate = wf.getframerate()
    recognizer = vosk.KaldiRecognizer(vosk_model, sample_rate)  # Converting the speech into text

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            pass

    result = recognizer.Result()
    return result

# Converting text into speech
def text_to_speech(text):
    st.info("Recognizing the text to speech...")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Setting up the chatbot for generating response
def get_chat_model_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat_model(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

#LLM model generation
def llm_model_for_intent_recognition(test_sentence):
    inputs = tokenizer(test_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_labels = torch.argmax(outputs.logits, dim=1)
    return int(predicted_labels)

def get_user_intent(text_input):
    # Using the LLM for intent recognition
    user_intent = llm_model_for_intent_recognition(text_input)

    # Map the predicted label to your specific intents
    if user_intent == 0:
        return "GET_BALANCE"
    elif user_intent == 1:
        return "GET_DUE"
    elif user_intent == 2:
        return "TRANSFER_MONEY"
    else:
        return "UNKNOWN"

#Extracting account number 
def extract_account_number(text):
    matches = re.findall(r'\b\d{6}\b', text)
    if matches:
        return matches[0]
    else:
        return None

#get account information.
def get_account_info(account_number):
    # load the JSON file to fetch your account details from your data source
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for account in data['accounts']:
        if account['account number'] == int(account_number):
            return {
                "name": account['name'],
                "account number": account['account number'],
                "balance": account['balance']
            }
    return None

# Reading the data from the JSON file
def read_data_from_json(voicefile):
    with open(voicefile, 'r') as f:
        data = json.load(f)
    return data

# Making changes in JSON file
def write_data_to_json(voicefile, data):
    with open(voicefile, 'w') as f:
        json.dump(data, f, indent=4)

# updating the account balance 
def update_account_balance(account_number, amount):
    with open(json_file_path, "r") as file:
        account_data = json.load(file)

    for account in account_data["accounts"]:
        if account["account number"] == account_number:
            # Updating the balance
            account["balance"] += amount
            break

    with open(json_file_path, "w") as file:        # Save the updated data back to the JSON file
        json.dump(account_data, file, indent=2)

# Transfering money
def transfer_money(sender_acc_no, receiver_acc_no, amount):
    data = read_data_from_json(json_file_path)
    sender_account = None
    receiver_account = None

    # Checking the sender and receiver account details
    for user in data['users']:
        for account in user['accounts']:
            if account['acc_no'] == str(receiver_acc_no):
                receiver_account = account
            elif account['acc_no'] == sender_acc_no:
                sender_account = account

    if sender_account is None or receiver_account is None:
        return False, "Sender or receiver account not found."

    if sender_account['balance'] < amount:
        return False, "Insufficient balance."

    sender_account['balance'] -= amount
    receiver_account['balance'] += amount

    write_data_to_json(json_file_path, data)

    return True, "Money Transfer is Succesful. Amount transferred by you is" + str(amount) + " to " + str(receiver_acc_no) + "."


# Checking the user inputs
def user_input():
    if st.session_state.get('values_entered', False): 
        return

    if st.button("Ask"):
        user_input = st.session_state['user_input']
        user_intent = get_user_intent(user_input)

        if user_intent == "TRANSFER_MONEY":
            st.info("Sure, let's initiate a money transfer.")
            st.session_state['expected_input'] = "TRANSFER_MONEY"
            text_to_speech("Kindly, please start a money transfer. Would you kindly supply the following information:")
            recipient_account = st.text_input("Recipient's account number:")
            transfer_amount = st.text_input("Transfer Amount:")
            if st.button("ok"):
                if recipient_account and transfer_amount:
                    sender_account_number = extract_account_number(user_input)

                    success, message = transfer_money(sender_account_number, recipient_account, float(transfer_amount))

                    if success:
                        st.success(message)
                        text_to_speech(f"Transfer successful. {message}")
                    else:
                        st.error(message)
                        text_to_speech(f"Transfer failed. {message}")

                else:
                    st.warning("Mention both recipient's account number and transfer amount.")
                    text_to_speech("Mention both recipient's account number and transfer amount.")

        elif user_intent == "GET_DUE":
            st.info("I'll verify your dues, of course.")
            account_number = st.text_input("Enter the account number:")
            if st.button("ok"):
                if account_number:
                    st.session_state['account_number'] = account_number
                    dues = get_account_info(account_number).get('dues', 0)
                    st.info(f"Your dues are {dues}$.")
                    text_to_speech(f"Your dues are {dues}$.")
                else:
                    st.warning("Account number cannot be extracted from the input. Please provide a valid account number.")
                    text_to_speech("Sorry, failed to process the account number. Please provide a valid account number.")

        elif user_intent == "GET_BALANCE":
            st.info("Okay, allow me to verify your balance.")
            account_number = st.text_input("Enter the account number:")
            if account_number:
                st.session_state['account_number'] = account_number
                balance = get_account_info(account_number).get('balance', 0)
                st.info(f"Your account balance is {balance}$.")
                text_to_speech(f"Your account balance is {balance}$.")
            else:
                st.warning("Account number cannot be extracted from the input. Please provide a valid account number.")
                text_to_speech("Sorry, failed to process the account number. Please provide a valid account number.")

        else:
            st.subheader("Chatbot Response:")
            response = get_chat_model_response(user_input)
            st.write(response)
            text_to_speech(response)

st.set_page_config(page_title="Banking Chatbot")
st.header("Banking Chatbot")

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="Welcome to Banking Chatbot. How may I help you?")
    ]

# Chitchatting
st.sidebar.title("User Information")
st.sidebar.info("Click the Audio Record button and speak with the chatbot.")
audio_data = st.sidebar.button("Record Audio")
if audio_data:
    wav_voicefile = record_audio()
    text_input = recognize_speech(wav_voicefile)
    st.session_state['user_input'] = text_input
    st.sidebar.text(f"User Input: {text_input}")
else:
    st.session_state['user_input'] = st.text_input("Your Message:")

user_input()