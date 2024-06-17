from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv


load_dotenv()

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")