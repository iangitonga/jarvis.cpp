from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# Size of the vocabulary which precludes special tokens e.g <|begin_of_text|>, etc
VOCAB_SIZE = 49152

def convert():
	with open("tokenizer.bin", "wb") as f:
		for i in range(VOCAB_SIZE):
			word = tokenizer.decode([i], clean_up_tokenization_spaces=False)
			word_byte = word.encode(encoding="utf-8")
			f.write(int.to_bytes(len(word_byte), length=4, byteorder="little"))
			f.write(word_byte)

convert()