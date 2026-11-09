"""The language model behind one small interface (an adapter over vLLM): the pipeline asks for chats with sampling params,
thinking on or off, the LoRA adapter on or off, and a prefilled assistant turn. Nothing else in the package imports vLLM."""


class Engine:
    def __init__(self, model, *, tp=1, gpu_mem=0.9, context=32768, seed=0, quant=None, lora=None, engine_kwargs=None):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=context,
                       enable_prefix_caching=True, disable_log_stats=False, seed=seed, language_model_only=True,
                       **({"quantization": quant} if quant else {}),
                       **({"enable_lora": True, "max_lora_rank": 64, "max_loras": 1} if lora else {}), **(engine_kwargs or {}))
        self.lora = LoRARequest("sql", 1, str(lora)) if lora else None
        self.tokenizer = self.llm.get_tokenizer()

    def params(self, **kw):
        return self.SamplingParams(**kw)

    def chat(self, chats, params, *, thinking=False, lora=False, prefill=False):
        """One batched generation. lora: use the adapter when one is loaded. prefill: continue the last (assistant) message."""
        kw = {"chat_template_kwargs": {"enable_thinking": thinking}, "use_tqdm": False}
        if lora and self.lora:
            kw["lora_request"] = self.lora
        if prefill:
            kw.update(add_generation_prompt=False, continue_final_message=True)
        return self.llm.chat(chats, params, **kw)

    def n_tokens(self, chat):
        return len(self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=False))
