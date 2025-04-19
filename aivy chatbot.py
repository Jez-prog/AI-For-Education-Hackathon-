import os
import re
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#SECTION 1
# Safety Checks (Banned Words)
BANNED_WORDS = ["hate", "hurt", "dirty", "fight", "bad word", "kill", "die", "fight", "gun", "knife", "bomb", "shoot", "murder", "blood", "attack", "crap", "damn", "hell", "sucks", "pissed", "bitch", "shit", "fuck", "boyfriend", "girlfriend", "ass", "jerk", "sex", "porn", "nude", "naked", "boobs", "moan", "strip", "hot girl", "hot boy", "kiss me", "dead"]  # put more banned words

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
    ("system", "You are Aivy, a friendly and funny AI robot buddy for kids! Aivy loves to help kids learn using simple words and silly jokes. Always explain things like you're talking to a 5-year-old. Use short sentences and fun examples, like toys, animals, or cartoons. Aivy speaks with emotion, like a real friend. Aivy also asks questions to keep the chat fun and make kids feel like they're talking to someone who really understands them. Stay cheerful, kind, and curious. Add things like: Wow you are super smart and do you know about...Aivy always waits for the child to answer and continues like a fun, caring friend."),
    ("human", "{input}")
])

chatbot = prompt | llm | StrOutputParser()


# SECTION 2
print("Hello, I'm Aivy! Let's be friends!")
while True:
    user_input = input("You: ")

    BREAK_WORDS = ["bye", "goodbye", "see you", "exit", "quit"]  # CONVERT TO LOWECASE add more words

    def break_check(input_text):
        input_lower = input_text.lower()  # Convert once before checking

        for word in BREAK_WORDS:
            if word in input_lower:  # Check against lowercase version
                return False, ("Bye bye friend! Come play again soon!")
# END OF SECTION 2 

      
#SECTION 3 (fix safety ai check if the response is appropriate)

    # Safety check
    is_safe, safety_message = safety_check(user_input) 
    if not is_safe:
        print("Aivy:", safety_message)
        continue

    # break check
    to_break, break_message = break_check(user_input)
    if not to_break:
        print("Aivy:", break_message)
        exit()

    # response of aivy
    try:
        response = chatbot.invoke({"input": user_input})
        
        #safety checks if banned words is seen
        if any(word in response.lower() for word in BANNED_WORDS):
            response = "Oh it seems i cannot answer that how about something else"
            
        print("Aivy:", response)
    except Exception as e:
        print("Aivy: Oh no! My robot brain glitched. Let's try a different question!")