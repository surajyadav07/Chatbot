# Chatbot
RAG based chatbot

AmpD Enertainer Manual — RAG Chatbot (Prototype)
This is a quick prototype chatbot that can answer questions about the Enertainer manual using Retrieval-Augmented Generation (RAG).

How to run it
Clone this repo.

Put the provided manual at:
data/enertainer_manual.pdf

Create and activate a virtual environment.
Install dependencies:
pip install -r requirements.txt

Indexing (one-time step)
Build the FAISS index from the manual:
python chatbot.py --rebuild

Chat (CLI)
Run the chatbot in your terminal:
python chatbot.py


Notes
To get better answers, set OPENAI_API_KEY in your environment.
Embeddings are from sentence-transformers, so no API cost there.

Index is saved at:
data/faiss_index.bin
data/index_meta.json

FAISS can be finicky to install depending on your OS. If you hit issues:
Pip: pip install faiss-cpu


Limitations
I didn’t use images from the manual. Handling them properly would need OCR or captioning models, which is doable but outside what I could cover in the time given.
There’s no conversational memory — the bot just answers one question at a time. Since the challenge only asked for a basic Q&A, I kept it simple.
Both of these could be added later if we extend.

