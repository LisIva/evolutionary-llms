import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from time import time
import creds


def generate_response(question, template, llm):
  prompt = PromptTemplate(template=template, input_variables=["question", "context"])
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response = llm_chain.run({"question":question})
  return response


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

login(token=creds.hf_key)

model_4bit = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")#, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# print("all done")


pipeline_inst = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=2500,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

t1 = time()
llm = HuggingFacePipeline(pipeline=pipeline_inst)

template = """<s>[INST] You are a respectful and helpful assistant, always be precise and politely answer questions in conversational english with minimum words.
Answer the question below from context below :
{question} [/INST] </s>
"""

print(generate_response("Name one president of America?", template, llm))

t2 = time()
print(f"Time spent, m: {(t2-t1) / 60:.2f}")

