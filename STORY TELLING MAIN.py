import random
import pyttsx3
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize text-to-speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Use a child-friendly voice
engine.setProperty('rate', 150)  # Slower speech for kids

def speak(text):
    print(f"StoryBot: {text}")
    engine.say(text)

# Story elements
heroes = ["curious AI bot", "friendly robot", "tech-savvy kid", "digital cat", "inventor teenager"]
settings = ["futuristic city", "underground tech lab", "space station", "robot school", "virtual reality world"]
goals = ["build a supercomputer", "stop a rogue virus", "win a coding challenge", "invent a teleportation device", "solve a space mystery"]

llm = Ollama(model="llama3.2:1b")

def generate_story():
    hero = random.choice(heroes)
    setting = random.choice(settings)
    goal = random.choice(goals)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are StoryBot, an AI that tells short, engaging stories for kids with multiple-choice questions. 
         Create a very short story about:
         Hero: {hero}
         Setting: {setting}
         Goal: {goal}
         
         Then ask ONE question with choice a,b,c,d. 
         Keep language simple for 5-8 year olds. Make it fun and educational!
         Format exactly like this:
         
         STORY: [Your story here]
         QUESTION: [Your question here]
         a: [choice a answer here]
         b: [choice b answer here]
         c: [choice c answer here]
         d: [choice d answer here]
        
         """)
    ])
    
    story_chain = prompt | llm | StrOutputParser()
    return story_chain.invoke({})

def check_answer(story_text, user_choice):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Given this story and question:
         {story_text}
         
         The user selected: {user_choice}
         
         Respond in 1-2 sentences telling them if they were correct or not, 
         and briefly explain why in simple terms a child would understand.
         If wrong, don't reveal the right answer yet - encourage them to try again!
         """),
    ])
    
    check_chain = prompt | llm | StrOutputParser()
    return check_chain.invoke({})

# Main interaction
speak("Hi! I'm StoryBot! Let me tell you a fun story!")

while True:
    # Generate and tell story
    story_text = generate_story()
    speak(story_text)
    
    # Extract question part for answer checking
    question_part = story_text.split("QUESTION:")[1] if "QUESTION:" in story_text else story_text
    
    # Get user answer
    while True:
        user_input = input(" ").lower().strip()
        
        if user_input == 'next':
            break
            
        if user_input in ['a', 'b', 'c', 'd']:
            feedback = check_answer(question_part, user_input)
            speak(feedback)
    
    speak("Great job! Want another story?")