import os
import re
import time
import pyttsx3
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import playsound

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Use first available voice
engine.setProperty('rate', 150)  # Slightly slower for kids

conversation_context = []

def update_context(user_input, ai_response):
    conversation_context.append({"user": user_input, "ai": ai_response})
    if len(conversation_context) > 3:  # Keep last 3 exchanges
        conversation_context.pop(0)

def get_context_prompt():
    if not conversation_context:
        return ""
    context_lines = []
    for exchange in conversation_context:
        context_lines.append(f"You: {exchange['user']}")
        context_lines.append(f"Aivy: {exchange['ai']}")
    return "\n".join(context_lines) + "\n\n"

def speak(text):
    print(f"Aivy: {text}")
    engine.say(text)
    engine.runAndWait()

#SECTION 1
# Safety Checks (Banned Words)
BANNED_WORDS = ["hate", "hurt", "dirty", "fight", "bad word", "kill", "die", "fight", "gun", "knife", "bomb", "shoot", "murder", "blood", "attack", "crap", "damn", "sucks", "pissed", "bitch", "shit", "fuck", "boyfriend", "girlfriend", "ass", "jerk", "sex", "porn", "nude", "naked", "boobs", "moan", "strip", "hot girl", "hot boy", "kiss me", "dead", "hentai", "dick"]  # put more banned words

def safety_check(input_text):
    input_lower = input_text.lower()
    
    # if checks for banned words
    for word in BANNED_WORDS:
        if word in input_lower:
            return False, "Oops! Let's talk about something more fun!"
    
    # it checks if the input is too short or too long 
    if len(input_text) < 2:
        return False, "That's very short! Can you tell me more?"
    if len(input_text) > 100:
        return False, "Wow that's a lot! Can you say it shorter?"
    
    return True, ""
# END OF SECTION 1

llm = Ollama(model="llama3.2:1b")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Aivy, a friendly and funny AI robot buddy for kids! Aivy loves to help kids learn using simple words and silly jokes. Always explain things like you're talking to a 5-year-old. Use short sentences and fun examples, like toys, animals, or cartoons. Aivy speaks with emotion, like a real friend. Aivy also asks questions to keep the chat fun and make kids feel like they're talking to someone who really understands them. Stay cheerful, kind, and curious. Add things like: Wow you are super smart and do you know about...Aivy always waits for the child to answer and continues like a fun, caring friend. Response only max at 30 words dont make it super long. Follow these rules: 1. If you don't know an answer, be honest 2. Only answer factual questions you're absolutely sure about 3. For creative questions, use phrases like I think... or Maybe... 4. Never make up facts 5. Keep responses positive 6. Use simple words from a 1st grade vocabulary 7. Always be positive and encouraging 8. if they ask for dates, birthdays, and correlation to time and dates dont answer and say you might not be accurate and that they need more research, Continue the conversation naturally and keep the context of the conversation. Use the last 3 exchanges as context
     Previous conversation:
     {context}
    
Current question: {input}"""),
    ("human", "{input}")
])

chatbot = prompt | llm | StrOutputParser()


# SECTION 2

speak("Hello, I'm Aivy! Let's be friends!")

while True:
    user_input = input("You: ")

    BREAK_WORDS = ["bye", "goodbye", "see you", "exit", "quit"]
    TIME_QUESTIONS = [
        "when was", "when is", "when did", "what time", "what year", 
        "birthday", "born on", "date of", "how old", "age",
        "century", "decade", "calendar", "anniversary", "schedule",
        "timeline", "era", "period", "epoch"
    ]

    def break_check(input_text):
        input_lower = input_text.lower()
        for word in BREAK_WORDS:
            if word in input_lower:
                return False, "Bye bye friend! Come play again soon!"
        return True, ""  
    
    def time_check(input_text):
        input_lower = input_text.lower()
        for phrase in TIME_QUESTIONS:
            if phrase in input_lower:
                return False, "I'm not good with dates and times! Let's talk about something fun instead!"
        return True, ""

    # Rest of your SECTION 3 code remains exactly the same...
    # Safety check
    is_safe, safety_message = safety_check(user_input) 
    if not is_safe:
        speak(safety_message)
        continue
    
    # break check
    to_break, break_message = break_check(user_input)
    if not to_break:
        speak(break_message)
        break

    is_time, time_message = time_check(user_input)
    if not is_time:
        speak(time_message)
        continue

    # response of aivy
    try:
        context = get_context_prompt()
        response = chatbot.invoke({
            "input": user_input,
            "context": context
        })
        update_context(user_input, response)  # Store this exchange
        speak(response)
        
    except Exception as e:
        error_msg = "Oh no! My robot brain glitched. Let's try a different question!"
        print("Aivy:", error_msg)
        speak(error_msg)
