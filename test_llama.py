import llama_cpp
import numpy as np

def test_qwen():
    llm = llama_cpp.Llama(model_path=r"F:\GlassNetwork_Models\qwen2.5-1.5b-instruct-q4_k_m.gguf", n_ctx=256, logits_all=True, verbose=False)
    print("vocab_size:", llm.n_vocab())
    first_byte_map = np.zeros(llm.n_vocab(), dtype=np.int32)
    for i in range(llm.n_vocab()):
        try:
            b = llm.detokenize([i])
            if len(b) > 0:
                first_byte_map[i] = b[0]
            else:
                first_byte_map[i] = -1
        except Exception:
            first_byte_map[i] = -1
    print("first byte map max:", first_byte_map.max())
    print("first byte map min:", first_byte_map.min())
    
    # eval
    tokens = llm.tokenize(b'hello', add_bos=False)
    llm.reset()
    llm.eval(tokens)
    logits = llm.scores[len(tokens)-1, :]
    print("logits shape:", logits.shape)
    
test_qwen()
