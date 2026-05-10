from dotenv import load_dotenv 
import os 
load_dotenv()
key=os.getenv("OPENAI_API_KEY")
if key:
    print(key[:8])
else:
    print("no key")