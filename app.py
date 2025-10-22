import gradio as gr
import random
from dataset_utils import load_dataset
from ai_model import build_embeddings, predict_condition
from nlp_utils import preprocess_input
import pandas as pd

# Load all datasets
data = load_dataset("Diseases_Symptoms.csv")
symptom_embeddings = build_embeddings(data["Symptoms"].tolist())

# Load additional datasets
try:
    descriptions_df = pd.read_csv("symptom_Description.csv")
    precautions_df = pd.read_csv("symptom_precaution.csv")
    severity_df = pd.read_csv("Symptom-severity.csv")
    
    # Create lookup dictionaries for faster access
    disease_descriptions = dict(zip(descriptions_df['Disease'], descriptions_df['Description']))
    disease_precautions = precautions_df.set_index('Disease').to_dict('index')
    symptom_severity = dict(zip(severity_df['Symptom'], severity_df['weight']))
except Exception as e:
    print(f"Warning: Could not load additional datasets: {e}")
    disease_descriptions = {}
    disease_precautions = {}
    symptom_severity = {}

def calculate_severity_score(symptoms_text):
    """Calculate overall severity based on mentioned symptoms"""
    if not symptom_severity:
        return None
    
    symptoms_lower = symptoms_text.lower()
    total_severity = 0
    matched_symptoms = 0
    
    for symptom, weight in symptom_severity.items():
        symptom_normalized = symptom.replace('_', ' ')
        if symptom_normalized in symptoms_lower:
            total_severity += weight
            matched_symptoms += 1
    
    if matched_symptoms > 0:
        return total_severity / matched_symptoms
    return None

def format_precautions(disease_name):
    """Format precautions in a user-friendly way"""
    if disease_name not in disease_precautions:
        return None
    
    precautions = disease_precautions[disease_name]
    precaution_list = []
    
    for i in range(1, 5):
        precaution_key = f'Precaution_{i}'
        if precaution_key in precautions and precautions[precaution_key]:
            precaution_text = str(precautions[precaution_key]).strip()
            if precaution_text and precaution_text.lower() != 'nan':
                precaution_list.append(precaution_text)
    
    return precaution_list

# --- Enhanced Conversational Templates ---
GREETINGS = [
    "Hi there! I'm here to help you understand your symptoms. ",
    "Hello! Thanks for reaching out. ",
    "Hey! I'm listening. ",
]

EMPATHY_PHRASES = {
    "low": ["I understand that can be uncomfortable. ", "That must be concerning. ", "I hear you. "],
    "medium": ["That sounds quite bothersome. ", "I can see why you're worried about this. ", "That must be affecting your day. "],
    "high": ["I'm really sorry you're dealing with this. ", "That sounds really difficult. ", "I can imagine how concerning this must be. "]
}

CONFIDENCE_INTROS = {
    "high": [
        "Based on what you've shared, it looks like this could be related to ",
        "From your symptoms, I'm thinking this might be ",
        "Your symptoms are pointing toward ",
    ],
    "medium": [
        "Your symptoms could possibly indicate ",
        "This might be related to ",
        "I'm seeing some patterns that suggest ",
    ],
    "low": [
        "I'm not entirely certain, but this could be ",
        "It's a bit tricky to say for sure, but possibly ",
        "Your symptoms are somewhat unclear, but they might suggest ",
    ]
}

FOLLOW_UP_QUESTIONS = [
    "How long have you been experiencing these symptoms?",
    "Have you noticed anything that makes it better or worse?",
    "Are there any other symptoms you haven't mentioned yet?",
    "Have you tried anything to relieve this?",
]

ACKNOWLEDGMENTS = [
    "I see. ", "Got it. ", "Okay. ", "Thanks for sharing that. ", 
    "I understand. ", "That helps. ", "Noted. "
]

TRANSITION_PHRASES = [
    "Let me think about this... ", "Hmm, interesting... ", 
    "Alright, based on that... ", "So... "
]

def extract_user_symptoms(user_input):
    """Extract key symptoms mentioned by user for personalized responses"""
    common_symptoms = ['headache', 'fever', 'cough', 'pain', 'nausea', 'tired', 
                       'fatigue', 'dizzy', 'sore throat', 'runny nose', 'chest pain',
                       'stomach', 'vomiting', 'diarrhea', 'rash', 'itching']
    mentioned = [s for s in common_symptoms if s in user_input.lower()]
    return mentioned

def generate_conversational_response(user_input, top_matches, is_first_message=False, mentioned_symptoms=None):
    """Generate a natural, human-like response with memory of symptoms"""
    
    if not top_matches:
        responses = [
            "Hmm, I'm having trouble matching your symptoms to anything specific in my knowledge base. Could you describe what you're feeling in a bit more detail? For example, where do you feel discomfort, when did it start, and how severe is it?",
            "I'm not quite getting a clear match here. Can you tell me more about what you're experiencing? Maybe describe the main symptom that's bothering you most?",
            "I want to help, but I need a bit more information. Could you elaborate on your symptoms? What exactly are you feeling?",
        ]
        return random.choice(responses)
    
    # Start building the response
    response_parts = []
    
    # Add acknowledgment for natural flow
    if not is_first_message and mentioned_symptoms:
        symptom_text = " and ".join(mentioned_symptoms[:2])
        acks = [
            f"So you're dealing with {symptom_text}. ",
            f"Okay, {symptom_text} - that's what's bothering you. ",
            f"Right, so {symptom_text} is what you're experiencing. "
        ]
        response_parts.append(random.choice(acks))
    elif not is_first_message:
        response_parts.append(random.choice(ACKNOWLEDGMENTS))
        response_parts.append(random.choice(TRANSITION_PHRASES))
    
    # Add greeting if first message
    if is_first_message:
        response_parts.append(random.choice(GREETINGS))
    
    # Get top match info
    top_match = top_matches[0]
    similarity = top_match['similarity']
    
    # Calculate severity if possible
    severity_score = calculate_severity_score(top_match['symptoms'])
    
    # Add empathy based on severity
    if severity_score and severity_score > 4:
        response_parts.append(random.choice(EMPATHY_PHRASES["high"]))
    elif similarity > 70:
        response_parts.append(random.choice(EMPATHY_PHRASES["low"]))
    elif similarity > 50:
        response_parts.append(random.choice(EMPATHY_PHRASES["medium"]))
    else:
        response_parts.append(random.choice(EMPATHY_PHRASES["high"]))
    
    # Add confidence-appropriate introduction
    if similarity > 70:
        intro = random.choice(CONFIDENCE_INTROS["high"])
    elif similarity > 50:
        intro = random.choice(CONFIDENCE_INTROS["medium"])
    else:
        intro = random.choice(CONFIDENCE_INTROS["low"])
    
    response_parts.append(intro)
    
    # Present the main condition
    if len(top_matches) == 1:
        response_parts.append(f"**{top_match['name']}**.")
    else:
        response_parts.append(f"**{top_match['name']}**, though it could also be {top_matches[1]['name'].lower()} or {top_matches[2]['name'].lower()}.")
    
    # Add disease description if available
    if top_match['name'] in disease_descriptions:
        description = disease_descriptions[top_match['name']]
        response_parts.append(f"\n\n**What is {top_match['name']}?**\n{description}")
    
    # Treatment information - Make it prominent
    response_parts.append("\n\n**What you can do about it:**")
    if top_match['treatment'] and top_match['treatment'] != 'N/A':
        treatment_intros = [
            "\nTypically, this is treated with",
            "\nCommon treatments include",
            "\nMost people manage this with",
            "\nDoctors usually recommend",
        ]
        response_parts.append(f"{random.choice(treatment_intros)} {top_match['treatment'].lower()}.")
    else:
        response_parts.append("\nUnfortunately, I don't have specific treatment information for this condition in my database. A healthcare provider can give you guidance on the best approach.")
    
    # Add precautions if available
    precautions = format_precautions(top_match['name'])
    if precautions:
        response_parts.append("\n\n**Important Precautions:**")
        for i, precaution in enumerate(precautions, 1):
            response_parts.append(f"\n{i}. {precaution.capitalize()}")
    
    # Add severity warning if high
    if severity_score and severity_score > 5:
        response_parts.append("\n\n⚠️ **Severity Note:** Based on your symptoms, this could be moderately to severely uncomfortable. If symptoms worsen or persist, seek medical attention promptly.")
    
    # Add other possible treatments if confidence is medium-high
    if similarity > 60 and len(top_matches) > 1:
        alt_treatments = []
        for match in top_matches[1:]:
            if match['treatment'] and match['treatment'] != 'N/A' and match['treatment'] != top_match['treatment']:
                alt_treatments.append(match['treatment'])
        
        if alt_treatments:
            response_parts.append(f"\n\nIf it turns out to be {top_matches[1]['name'].lower()}, treatment might involve {alt_treatments[0].lower()} instead.")
    
    # Add details naturally
    response_parts.append(f"\n\n**Why I think it might be this:**\nPeople with {top_match['name'].lower()} typically experience: {top_match['symptoms'].lower()}.")
    
    # Important attributes in conversational form
    important_notes = []
    if top_match['contagious']:
        important_notes.append("it can spread to others, so take precautions")
    if top_match['chronic']:
        important_notes.append("it tends to be ongoing and may need long-term management")
    
    if important_notes:
        response_parts.append(f" Keep in mind that {' and '.join(important_notes)}.")
    
    # Add follow-up or closing based on confidence
    if similarity < 60:
        response_parts.append(f"\n\n{random.choice(FOLLOW_UP_QUESTIONS)}")
    else:
        closings = [
            "\n\nDoes this sound like what you're experiencing?",
            "\n\nDo any of these other symptoms match what you're feeling?",
            "\n\nIs this lining up with what you're going through?",
        ]
        response_parts.append(random.choice(closings))
    
    # Always add medical disclaimer
    disclaimer_options = [
        "\n\n⚠️ *Remember, I'm here to provide information, but you should definitely consult with a healthcare professional for a proper diagnosis and treatment plan.*",
        "\n\n⚠️ *This is just preliminary guidance – please see a doctor to get properly checked out.*",
        "\n\n⚠️ *I'd really recommend talking to a healthcare provider about this to be sure.*",
    ]
    response_parts.append(random.choice(disclaimer_options))
    
    return "".join(response_parts)

# --- Chat history with context ---
chat_history = []
message_count = 0
last_diagnosis = None
conversation_context = {
    "symptoms_mentioned": [],
    "user_confirmed": False,
    "asked_duration": False,
}

def chatbot_response(user_input, chat_history):
    global message_count, last_diagnosis, conversation_context
    message_count += 1
    
    user_input_lower = user_input.lower().strip()
    
    # Handle greetings
    greetings_list = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'hi there', 'hello there']
    if user_input_lower in greetings_list:
        responses = [
            "Hello! I'm here to help you understand your symptoms. What's been bothering you?",
            "Hi there! How are you feeling today? Tell me about your symptoms.",
            "Hey! I'm all ears. What symptoms are you experiencing?",
        ]
        response = random.choice(responses)
        chat_history.append((user_input, response))
        return "", chat_history
    
    # Handle thank you responses
    thanks_keywords = ['thank', 'thanks', 'thank you', 'appreciate', 'helpful']
    if any(keyword in user_input_lower for keyword in thanks_keywords):
        thank_responses = [
            "You're very welcome! I hope you feel better soon. Don't hesitate to come back if you have more questions. Take care! 😊",
            "Happy to help! Please take care of yourself and see a doctor if things don't improve. Feel better soon!",
            "Glad I could assist! Remember to seek professional medical care if needed. Wishing you a speedy recovery! 🌟",
        ]
        response = random.choice(thank_responses)
        chat_history.append((user_input, response))
        last_diagnosis = None
        conversation_context = {"symptoms_mentioned": [], "user_confirmed": False, "asked_duration": False}
        return "", chat_history
    
    # Handle goodbye
    goodbye_keywords = ['bye', 'goodbye', 'see you', 'gotta go', 'talk later', 'gtg']
    if any(keyword in user_input_lower for keyword in goodbye_keywords):
        goodbye_responses = [
            "Take care! I hope you feel better soon. Come back anytime you need help! 👋",
            "Goodbye! Please take care of yourself and don't hesitate to reach out again. Feel better!",
            "See you! Wishing you good health. Remember to consult a doctor if things get worse! 🌈",
        ]
        response = random.choice(goodbye_responses)
        chat_history.append((user_input, response))
        last_diagnosis = None
        conversation_context = {"symptoms_mentioned": [], "user_confirmed": False, "asked_duration": False}
        return "", chat_history
    
    # Handle Yes/No responses for follow-up
    yes_responses = ['yes', 'yeah', 'yep', 'yup', 'correct', 'right', 'exactly', 'that\'s right', 'yes it does', 'sounds right', 'matches']
    no_responses = ['no', 'nope', 'nah', 'not really', 'no it doesn\'t', 'that\'s not it', 'doesn\'t match', 'wrong']
    
    if last_diagnosis and (user_input_lower in yes_responses or any(phrase in user_input_lower for phrase in yes_responses)):
        condition_name = last_diagnosis['name']
        treatment = last_diagnosis['treatment']
        conversation_context["user_confirmed"] = True
        
        confirmations = [
            f"Okay, so it sounds like **{condition_name}** matches what you're going through. ",
            f"Got it! So we're looking at **{condition_name}** here. ",
            f"Alright, **{condition_name}** seems to be what's happening. ",
            f"Perfect, so **{condition_name}** is what we're dealing with. ",
        ]
        
        response_parts = [random.choice(confirmations)]
        
        # Provide treatment
        if treatment and treatment != 'N/A':
            next_steps = [
                f"Here's what I'd suggest: {treatment.lower()}. ",
                f"The best approach would be: {treatment.lower()}. ",
                f"You'll want to focus on: {treatment.lower()}. ",
            ]
            response_parts.append(random.choice(next_steps))
        
        # Add precautions if available
        precautions = format_precautions(condition_name)
        if precautions:
            response_parts.append("\n\n**Don't forget these precautions:**")
            for i, precaution in enumerate(precautions, 1):
                response_parts.append(f"\n{i}. {precaution.capitalize()}")
        
        # Add timeline questions if not asked yet
        if not conversation_context["asked_duration"]:
            follow_ups = [
                "\n\nHow long have you been dealing with these symptoms? That can help determine if you need to see a doctor urgently.",
                "\n\nWhen did this start? If it's been going on for a while, definitely consider seeing a healthcare provider.",
                "\n\nHave the symptoms gotten worse over time, or have they stayed about the same?",
            ]
            response_parts.append(random.choice(follow_ups))
            conversation_context["asked_duration"] = True
        else:
            response_parts.append("\n\n**Is there anything else you'd like to know, or any other symptoms bothering you?**")
        
        response = "".join(response_parts)
        chat_history.append((user_input, response))
        return "", chat_history
    
    elif last_diagnosis and (user_input_lower in no_responses or any(phrase in user_input_lower for phrase in no_responses)):
        clarifications = [
            "Okay, let me try again. Can you tell me more specifically what symptoms you're experiencing? Maybe describe the main thing that's bothering you most?",
            "No problem! Let's dig a bit deeper. What exactly are you feeling? Be as specific as you can about your symptoms.",
            "Alright, let's reassess. Can you describe your symptoms in more detail? What's the primary discomfort you're experiencing?",
            "Got it, let me reconsider. What other symptoms do you have that I might have missed? Or which symptoms are the most prominent?",
        ]
        
        response = random.choice(clarifications)
        last_diagnosis = None
        conversation_context = {"symptoms_mentioned": [], "user_confirmed": False, "asked_duration": False}
        chat_history.append((user_input, response))
        return "", chat_history
    
    # Handle duration questions
    duration_keywords = ['day', 'days', 'week', 'weeks', 'month', 'months', 'hour', 'hours', 'since', 'ago', 'started', 'yesterday', 'today', 'this morning', 'last night']
    if last_diagnosis and any(keyword in user_input_lower for keyword in duration_keywords):
        urgent_keywords = ['week', 'weeks', 'month', 'months', 'getting worse', 'severe', 'really bad']
        is_urgent = any(keyword in user_input_lower for keyword in urgent_keywords)
        
        if is_urgent:
            duration_responses = [
                f"Thanks for letting me know. Based on that timeline, I'd definitely recommend seeing a healthcare provider soon to get properly evaluated. They can confirm if it's **{last_diagnosis['name']}** and prescribe the right treatment. This has been going on long enough that professional care is important.",
                f"I appreciate that information. Given how long this has been going on, it's really important to get checked out by a doctor. They can properly diagnose the **{last_diagnosis['name']}** and make sure you get the right care before it gets worse.",
            ]
        else:
            duration_responses = [
                f"Got it. Since it's relatively recent, the treatment I mentioned earlier should help. But if it doesn't improve in a few days or gets worse, definitely see a doctor to confirm it's **{last_diagnosis['name']}**.",
                f"Okay, that's still pretty fresh. Try the treatments I suggested, and monitor how you're feeling. If there's no improvement or it worsens, get medical attention to be safe.",
            ]
        
        response = random.choice(duration_responses)
        response += "\n\n**Is there anything else you'd like to know about your symptoms or condition?**"
        chat_history.append((user_input, response))
        return "", chat_history
    
    # Handle "tell me more" requests
    more_info_keywords = ['tell me more', 'more info', 'more information', 'what else', 'explain', 'details', 'elaborate', 'can you tell me']
    if last_diagnosis and any(keyword in user_input_lower for keyword in more_info_keywords):
        condition_name = last_diagnosis['name']
        symptoms = last_diagnosis['symptoms']
        contagious = last_diagnosis['contagious']
        chronic = last_diagnosis['chronic']
        
        response_parts = [f"Sure! Here's more about **{condition_name}**:\n\n"]
        
        # Add description if available
        if condition_name in disease_descriptions:
            response_parts.append(f"**Description:** {disease_descriptions[condition_name]}\n\n")
        
        response_parts.append(f"**Common symptoms include:** {symptoms.lower()}\n\n")
        
        if contagious:
            response_parts.append("**Contagiousness:** Yes, this condition can spread to others. You should take precautions like washing hands frequently, avoiding close contact, and staying home if possible.\n\n")
        else:
            response_parts.append("**Contagiousness:** No, this isn't contagious, so you don't need to worry about spreading it to others.\n\n")
        
        if chronic:
            response_parts.append("**Duration:** This tends to be a chronic (long-term) condition that may require ongoing management and regular check-ups.\n\n")
        else:
            response_parts.append("**Duration:** This is typically an acute condition that should resolve with proper treatment within days to weeks.\n\n")
        
        # Add precautions
        precautions = format_precautions(condition_name)
        if precautions:
            response_parts.append("**Precautions to take:**")
            for i, precaution in enumerate(precautions, 1):
                response_parts.append(f"\n{i}. {precaution.capitalize()}")
            response_parts.append("\n\n")
        
        response_parts.append("**What else would you like to know? Feel free to ask anything!**")
        
        response = "".join(response_parts)
        chat_history.append((user_input, response))
        return "", chat_history
    
    # Handle vague responses
    if len(user_input.split()) < 3 and last_diagnosis and not conversation_context["user_confirmed"]:
        prompts = [
            "Could you give me a bit more detail? The more specific you are, the better I can help!",
            "I need a little more information to assist you properly. Can you describe what you're feeling in more detail?",
            "That's pretty brief! Can you elaborate on your symptoms so I can give you better guidance?",
        ]
        response = random.choice(prompts)
        chat_history.append((user_input, response))
        return "", chat_history
    
    # Extract symptoms for context
    mentioned_symptoms = extract_user_symptoms(user_input)
    if mentioned_symptoms:
        conversation_context["symptoms_mentioned"].extend(mentioned_symptoms)
        conversation_context["symptoms_mentioned"] = list(set(conversation_context["symptoms_mentioned"]))
    
    # Process new symptoms
    processed_input = preprocess_input(user_input)
    top_matches = predict_condition(processed_input, data, symptom_embeddings, top_k=3)
    
    # Generate conversational response
    is_first = message_count == 1
    response = generate_conversational_response(user_input, top_matches, is_first, mentioned_symptoms)
    
    # Store the diagnosis for follow-up
    if top_matches:
        last_diagnosis = top_matches[0]
    
    # Append to history
    chat_history.append((user_input, response))
    
    return "", chat_history

# --- Gradio UI ---
with gr.Blocks(css="""
    .chat-message.user {background-color: #DCF8C6; padding: 8px; border-radius: 10px; margin:5px 0;}
    .chat-message.bot {background-color: #F1F0F0; padding: 8px; border-radius: 10px; margin:5px 0;}
    .chat-container {background-color: #E5DDD5; padding: 15px; border-radius: 15px; max-width:600px; margin:auto;}
    .send-btn {background-color:#128C7E; color:white; border:none; padding:10px 15px; border-radius:8px;}
    .send-btn:hover {background-color:#075E54; cursor:pointer;}
""") as demo:

    gr.Markdown("<h2 style='text-align:center; color:#128C7E;'>CarePlus AI – Smart Symptom Chatbot</h2>")
    gr.Markdown("<p style='text-align:center; color:#666;'>Chat naturally about your symptoms, and I'll help you understand what might be going on.</p>")
    
    with gr.Column(elem_classes="chat-container"):
        chatbot = gr.Chatbot()
        with gr.Row():
            msg = gr.Textbox(placeholder="Type your symptoms here...", show_label=False)
            send_btn = gr.Button("➤", elem_classes="send-btn")
        clear_btn = gr.Button("Clear Chat")

    def clear_chat():
        global message_count, last_diagnosis, conversation_context
        message_count = 0
        last_diagnosis = None
        conversation_context = {"symptoms_mentioned": [], "user_confirmed": False, "asked_duration": False}
        return []

    # Connect send button and enter key
    msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot])
    send_btn.click(chatbot_response, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_chat, None, chatbot)

demo.launch()