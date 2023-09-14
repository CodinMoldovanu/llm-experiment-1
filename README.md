# llm-experiment-1

### What's this?
This is a semi-first foray into LLMs and modern tooling. Its main purpose is retrieving some data from an URL, parsing it, storing it into a vector database, then using it as a context to be passed to a Llama2 model running locally to produce a desired output (initial desire was related to reading headlines).

Afterwards the output is sanitized, tokenized into sentences and passed on to Bark which is another locally hosted model this time for Text-To-Speech (TTS), and finally the generated speech is saved as a .wav file.

### How to run
```
virtualenv venv && 
pip install -r requirements.txt
```
Change the model path on line 49 to the path of your model, you can find `llama-2-7b-chat.Q4_0.gguf` on HuggingFace after submitting an access request to Meta (or use other llama2-7b-chat.Q4_0.gguf from other providers).