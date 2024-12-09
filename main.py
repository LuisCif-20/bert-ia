from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def get_response(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    responses = {
        0: "Hola, ¿cómo estás?",
        1: "¡Buenos días! ¿En qué ciudad te encuentras?",
        2: "Puedo ayudarte con varias cosas. ¿Qué necesitas saber?",
        3: "Soy Copilot, tu asistente virtual.",
        4: "Mi nombre no es importante, pero estoy aquí para ayudarte.",
        5: "Ahora mismo no tengo un reloj, pero seguro que sabes la hora.",
        6: "De nada, ¡hasta luego!"
    }
    
    response = responses.get(prediction, "No estoy seguro de cómo responder a eso.")
    return response

if __name__ == "__main__":
    tokenizer, model = load_model('models/chatbot_model')
    while True:
        input_text = input("Usuario: ")
        response = get_response(model, tokenizer, input_text)
        print(f"Chatbot: {response}")
