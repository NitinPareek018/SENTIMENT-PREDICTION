import pandas as pd
import csv
import requests
from tqdm import tqdm
 
# Ollama model you want to use
OLLAMA_MODEL = "llama3"  # or "mistral", "gemma", etc.
OLLAMA_URL = "http://localhost:11434/api/generate"
 
# Load CSV with a 'message' column
df = pd.read_csv("FreeTextData1.csv")




OUTPUT_CSV = "messages_with_sentimenit_ollama.csv"


with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['freetextid','messages', 'sentiment_category'])
 
# Classification prompt
def classify_with_ollama(message):
    prompt = f"""
You are a sentiment and intent classification assistant. 

Your task is to classify each **single message** (which can be a number, review, question, or complaint) into **only one** of the following categories:

- Excellent  
- Very Good  
- Good  
- Average  
- Poor  
- Inquiry  
- Car Not Recieved  
- Vehicle Sold  
- Mobile Number Wrong  
- Bill Issue  
- Others  

---

### Classification Rules (Follow strictly and in this order):

1. **Self-rating statements** (e.g., “I would say Good” or “Very Good service”) → Use the **stated sentiment**.

2. **Numeric ratings (1–10 only)**:
   - 9 or 10 → Excellent  
   - 7 or 8 → Very Good
   - 5 or 6 → Good   
   - 4 or 5 → Average  
   - 1 to 4 → Poor 
   - 0  → Poor
   - Greater than 10 → Others

3. **If the message is a question or request for info (status, location, price, when, where)** → Inquiry

4. **If message contains issues like "car not received","Service not complete now how to rate", "delivery pending", "gaadi toh aane do", "didn’t get car", "not delivered yet"** → Car Not Recieved

5. **If message says vehicle is not theirs, belongs to someone else, sold it, or "no car with me", "yeh meri gadi nahi hai"** → Vehicle Sold

6. **If it mentions 'wrong mobile number', 'this is not my number', 'don’t send messages', 'not my name'** → Mobile Number Wrong

7. **If the message is about ‘bill not received’, ‘bill issue’, or ‘bill pending’** → Bill Issue

8. **If the message includes strongly positive informal words like:**  
   - "badhiya", "badiya", "badia", "sahi", "acha", "gajab"  
     → classify as **Good**  
   - If these are preceded by "bahut", "bhot", "bhut"  
     → classify as **Excellent**

9. **If the message includes negative informal words like:**  
   - "kharab", "khrb", "khrab", "ganda", "gandi", "gndi", "ghatiya", "ghatia"  
     → classify as **Poor**

10. **Neutral or mixed tone reviews** → Average

11. **URLs, date mentions (like '10-oct', '5-nov', 'october'), numbers like (.08) or (,08), regional scripts, gibberish (e.g., “ðŸ‘”)** → Others

12. **Simple acknowledgments like 'ok', 'okay', 'yes', 'no', 'thanks', 'thank you', or 'Hi' , 'Hello'  random characters** → Others
13 . If word is second service or Have a great day  or not releated to service marks as → Others
---          

 
Message: "{message}"
 
Respond with only one category from this list.
"""
 
 
 
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
 
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        category = response.json()['response'].strip()
        print(f"Classified '{message}' as '{category}'")
        return category
    except Exception as e:
        print(f"Error: {e}")
        return "Others"
with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        msg_id = row['freetextid']
        msg = str(row['messages'])
        if msg_id > 1092866:
            category = classify_with_ollama(msg)
            writer.writerow([msg_id ,msg, category])
            file.flush()  # ensures immediate write to disk 
# Apply to all rows
# tqdm.pandas()
# df['sentiment_category'] = df['messages'].progress_apply(classify_with_ollama)
 
# # Save to new CSV
# df.to_csv("messages_with_sentiment_ollama.csv", index=False)
# print("Done! Saved sentiment results.")











