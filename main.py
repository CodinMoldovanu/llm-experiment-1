from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from chromadb.utils import embedding_functions
from langchain.embeddings import GPT4AllEmbeddings
from transformers import AutoProcessor, AutoModel
import scipy
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import time
import nltk
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
import numpy as np




nltk.download('punkt')
loader = WebBaseLoader("https://www.digi24.ro")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits,embedding=GPT4AllEmbeddings())

question = "What are the news in Romania today 14 September 2023? Limit to 4."
docs = vectorstore.similarity_search(question)



n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/codinm/Downloads/llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
res = qa_chain({"query": question})
print(res['result'])

time.sleep(15)

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

inputs = processor(
    text=[res['result']],
    return_tensors="pt",
)

sentences = nltk.sent_tokenize(res['result'].replace("\n", " ").strip())
SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))
pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]

# Audio(np.concatenate(pieces), rate=SAMPLE_RATE)

speech_values = model.generate(**inputs, do_sample=True)
scipy.io.wavfile.write("news1.wav", rate=SAMPLE_RATE, data=np.concatenate(pieces))