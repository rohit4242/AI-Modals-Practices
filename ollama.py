from typing import Optional, Mapping, List, Any

import modal
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import huggingface_pipeline
from transformers import pipeline
tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
)

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt_llm==0.10.0.dev2024042300",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

MODEL_DIR = "/root/model/model_input"
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"



def download_model():
    import os
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Set your Hugging Face access token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'
    
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        token=os.environ['HUGGING_FACE_HUB_TOKEN']  # Use the token for authentication
    )
    move_cache()


MINUTES = 60  # seconds
tensorrt_image = (  # update the image by downloading the model we're using
    tensorrt_image.pip_install(  # add utilities for downloading the model
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "requests~=2.31.0",
        "crewai"
    )
    .env(  # hf-transfer: faster downloads, but fewer comforts
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )
    .run_function(  # download the model
        download_model,
        timeout=20 * MINUTES,
    )
)


GIT_HASH = "71d8d4d3dc655671f32535d6d2b60cab87f36e87"
CHECKPOINT_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_HASH}/examples/llama/convert_checkpoint.py"

N_GPUS = 1  # Heads up: this example has not yet been tested with multiple GPUs
GPU_CONFIG = modal.gpu.H100(count=N_GPUS)


DTYPE = "float16"

# We put that all together with another invocation of `.run_commands`.

CKPT_DIR = "/root/model/model_ckpt"
tensorrt_image = (  # update the image by converting the model to TensorRT format
    tensorrt_image.run_commands(  # takes ~5 minutes
        [
            f"wget {CHECKPOINT_SCRIPT_URL} -O /root/convert_checkpoint.py",
            f"python /root/convert_checkpoint.py --model_dir={MODEL_DIR} --output_dir={CKPT_DIR}"
            + f" --tp_size={N_GPUS} --dtype={DTYPE}",
        ],
        gpu=GPU_CONFIG,  # GPU must be present to load tensorrt_llm
    )
)


MAX_INPUT_LEN, MAX_OUTPUT_LEN = 256, 256
MAX_BATCH_SIZE = (
    128  # better throughput at larger batch sizes, limited by GPU RAM
)
ENGINE_DIR = "/root/model/model_output"

SIZE_ARGS = f"--max_batch_size={MAX_BATCH_SIZE} --max_input_len={MAX_INPUT_LEN} --max_output_len={MAX_OUTPUT_LEN}"


PLUGIN_ARGS = f"--gemm_plugin={DTYPE} --gpt_attention_plugin={DTYPE}"


tensorrt_image = (  # update the image by building the TensorRT engine
    tensorrt_image.run_commands(  # takes ~5 minutes
        [
            f"trtllm-build --checkpoint_dir {CKPT_DIR} --output_dir {ENGINE_DIR}"
            + f" --tp_size={N_GPUS} --workers={N_GPUS}"
            + f" {SIZE_ARGS}"
            + f" {PLUGIN_ARGS}"
        ],
        gpu=GPU_CONFIG,  # TRT-LLM compilation is GPU-specific, so make sure this matches production!
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
)

app = modal.App(
    f"different-{MODEL_ID.split('/')[-1]}", image=tensorrt_image
)

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.cls(
    gpu=GPU_CONFIG,
    container_idle_timeout=5 * MINUTES,
)




class Model:
    @modal.enter()
    def load(self):
        """Loads the TRT-LLM engine and configures our tokenizer.

        The @enter decorator ensures that it runs only once per container, when it starts."""
        import time

        print(
            f"{COLOR['HEADER']}🥶 Cold boot: spinning up TRT-LLM engine{COLOR['ENDC']}"
        )
        self.init_start = time.monotonic_ns()

        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # LLaMA models do not have a padding token, so we use the EOS token
        self.tokenizer.add_special_tokens(
            {"pad_token": self.tokenizer.eos_token}
        )
        # and then we add it from the left, to minimize impact on the output
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

        self.init_duration_s = (time.monotonic_ns() - self.init_start) / 1e9
        print(
            f"{COLOR['HEADER']}🚀 Cold boot finished in {self.init_duration_s}s{COLOR['ENDC']}"
        )

    @modal.method()
    def generate(self, prompts: list[str], settings=None):
        """Generate responses to a batch of prompts, optionally with custom inference settings."""
        import time

        if settings is None or not settings:
            settings = dict(
                temperature=0.1,  # temperature 0 not allowed, so we set top_k to 1 to get the same effect
                top_k=1,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings[
            "max_new_tokens"
        ] = MAX_OUTPUT_LEN  # exceeding this will raise an error
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id

        num_prompts = len(prompts)

        if num_prompts > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {num_prompts} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        print(
            f"{COLOR['HEADER']}🚀 Generating completions for batch of size {num_prompts}...{COLOR['ENDC']}"
        )
        start = time.monotonic_ns()

        parsed_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

        print(
            f"{COLOR['HEADER']}Parsed prompts:{COLOR['ENDC']}",
            *parsed_prompts,
            sep="\n\t",
        )

        inputs_t = self.tokenizer(
            parsed_prompts, return_tensors="pt", padding=True, truncation=False
        )["input_ids"]

        print(
            f"{COLOR['HEADER']}Input tensors:{COLOR['ENDC']}", inputs_t[:, :8]
        )

        outputs_t = self.model.generate(inputs_t, **settings)

        outputs_text = self.tokenizer.batch_decode(
            outputs_t[:, 0]
        )  # only one output per input, so we index with 0

        responses = [
            extract_assistant_response(output_text)
            for output_text in outputs_text
        ]
        duration_s = (time.monotonic_ns() - start) / 1e9

        num_tokens = sum(
            map(lambda r: len(self.tokenizer.encode(r)), responses)
        )

        for prompt, response in zip(prompts, responses):
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{prompt}",
                f"\n{COLOR['BLUE']}{response}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)  # to avoid log truncation

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_ID} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second for batch of size {num_prompts} on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return responses
    
    @modal.method()
    def generate_response(self, prompt: str, settings=None):
        """Generate a single response to a prompt, for use by CrewAI agents."""
        if settings and 'stop' in settings:
            # Handle stop words if needed
            pass
        responses = self.model.generate([prompt], settings=settings)
        return responses[0] if responses else ""

# Define the request models
class TextRequest(BaseModel):
    text: str

class TensorRTLLMWrapper:
    def __init__(self, model: Model, **kwargs):
        self.model = model
        self.additional_params = kwargs

    def __call__(self, prompt: str, **kwargs) -> str:
        # Merge additional_params with kwargs, with kwargs taking precedence
        params = {**self.additional_params, **kwargs}
        return self.model.generate_response.remote(prompt, settings=params)

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        params = {**self.additional_params, **kwargs}
        return self.model.generate.remote(prompts, settings=params)

    def get_num_tokens(self, text: str) -> int:
        # Implement token counting logic here if needed
        # For now, we'll use a simple approximation
        return len(text.split())

    def get_num_tokens_from_messages(self, messages: List[Mapping[str, Any]]) -> int:
        return sum(self.get_num_tokens(m['content']) for m in messages)

    def invoke(self, input: Any, config: Optional[Any] = None) -> Any:
        if isinstance(input, str):
            return self(input)
        elif isinstance(input, list):
            return self.generate(input)
        else:
            raise ValueError("Unsupported input type")

    def stream(self, input: Any, config: Optional[Any] = None) -> Any:
        # Implement streaming if supported, otherwise raise an error
        raise NotImplementedError("Streaming is not supported for this model")

    def bind(self, **kwargs: Any) -> 'TensorRTLLMWrapper':
        # Create a new instance with the same model and updated kwargs
        new_kwargs = {**self.additional_params, **kwargs}
        return self.__class__(self.model, **new_kwargs)
    

class SummaryAgent(Agent):
    def __init__(self, llm: TensorRTLLMWrapper):
        super().__init__(
            name="Summary Agent",
            role="Summary Creator",
            goal="Provide concise and accurate summaries of given text",
            backstory="I am an AI agent specialized in creating summaries.",
            allow_delegation=False,
            llm=llm
        )

class ImportantQuestionAgent(Agent):
    def __init__(self, llm: TensorRTLLMWrapper):
        super().__init__(
            name="Important Question Agent",
            role="Question Extractor",
            goal="Identify and formulate the most important questions from given text",
            backstory="I am an AI agent focused on extracting key questions from content.",
            allow_delegation=False,
            llm=llm
        )

class QuizGeneratorAgent(Agent):
    def __init__(self, llm: TensorRTLLMWrapper):
        super().__init__(
            name="Quiz Generator Agent",
            role="Quiz Creator",
            goal="Create engaging and informative quizzes based on given content",
            backstory="I am an AI agent specialized in generating quizzes to test knowledge.",
            allow_delegation=False,
            llm=llm
        )

def create_summary_task(text, llm):
    return Task(
        description=f"Summarize the following text:\n{text}",
        expected_output="A concise summary of the given text",
        agent=SummaryAgent(llm),
    )

def create_important_questions_task(text, llm):
    return Task(
        description=f"What are the 3 most important questions raised by this text:\n{text}",
        expected_output="A list of 3 important questions raised by the given text",
        agent=ImportantQuestionAgent(llm),
    )

def create_quiz_task(text, llm):
    return Task(
        description=f"Generate a 5-question quiz based on this text:\n{text}",
        expected_output="A 5-question quiz based on the given text",
        agent=QuizGeneratorAgent(llm),
    )

@app.local_entrypoint()
def main():
    text = "Your sample text here.... This can be a longer piece of content that you want to process."

    model = Model()
    llm_wrapper = TensorRTLLMWrapper(model)

    summary_crew = Crew(
        agents=[SummaryAgent(llm_wrapper)],
        tasks=[create_summary_task(text, llm_wrapper)],
        process=Process.sequential,
    )

    questions_crew = Crew(
        agents=[ImportantQuestionAgent(llm_wrapper)],
        tasks=[create_important_questions_task(text, llm_wrapper)],
        process=Process.sequential,
    )

    quiz_crew = Crew(
        agents=[QuizGeneratorAgent(llm_wrapper)],
        tasks=[create_quiz_task(text, llm_wrapper)],
        process=Process.sequential,
    )

    print("Summary:", summary_crew.kickoff())
    print("Important Questions:", questions_crew.kickoff())
    print("Quiz:", quiz_crew.kickoff())

web_image = modal.Image.debian_slim(python_version="3.10")

# From there, we can take the same remote generation logic we used in `main`
# and serve it with only a few more lines of code.
model = Model()
llm_wrapper = TensorRTLLMWrapper(model)

@web_app.post("/summarize")
async def summarize(request: TextRequest):
    task = create_summary_task(request.text, llm_wrapper)
    crew = Crew(
        agents=[task.agent],
        tasks=[task],
        process=Process.sequential
    )
    result = crew.kickoff()
    return {"summary": result}

@web_app.post("/important-questions")
async def important_questions(request: TextRequest):
    task = create_important_questions_task(request.text, llm_wrapper)
    crew = Crew(
        agents=[task.agent],
        tasks=[task],
        process=Process.sequential
    )
    result = crew.kickoff()
    return {"important_questions": result}

@web_app.post("/generate-quiz")
async def generate_quiz(request: TextRequest):
    task = create_quiz_task(request.text, llm_wrapper)
    crew = Crew(
        agents=[task.agent],
        tasks=[task],
        process=Process.sequential
    )
    result = crew.kickoff()
    return {"quiz": result}

class GenerateRequest(BaseModel):
    prompts: list[str]
    settings: Optional[dict] = None



@app.function(allow_concurrent_inputs=4)
@modal.asgi_app()
def entrypoint():
    return web_app


@app.function(image=web_image)
@modal.web_endpoint(
    method="POST", label=f"{MODEL_ID.lower().split('/')[-1]}-web-dev", docs=True
)
def generate_web(data: GenerateRequest) -> list[str]:
    """Generate responses to a batch of prompts, optionally with custom inference settings."""
    return Model.generate.remote(data.prompts, settings=None)



COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


def extract_assistant_response(output_text):
    """Model-specific code to extract model responses.

    See this doc for LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/."""
    # Split the output text by the assistant header token
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")

    if len(parts) > 1:
        # Join the parts after the first occurrence of the assistant header token
        response = parts[1].split("<|eot_id|>")[0].strip()

        # Remove any remaining special tokens and whitespace
        response = response.replace("<|eot_id|>", "").strip()

        return response
    else:
        return output_text