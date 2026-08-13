import gradio as gr
import os
import time
import gc
import mimetypes
import base64
import PyPDF2
import io

try:
    from llama_cpp import Llama
    print("llama-cpp-python available!")
except ImportError:
    print("Installing llama-cpp-python...")
    import subprocess
    subprocess.run(["pip", "install", "llama-cpp-python==0.2.90", "--no-cache-dir"], check=False)
    try:
        from llama_cpp import Llama
        print("llama-cpp-python installed successfully!")
    except ImportError:
        print("Failed to install llama-cpp-python")
        exit(1)

try:
    import PyPDF2
    print("PyPDF2 available!")
except ImportError:
    print("Installing PyPDF2...")
    import subprocess
    subprocess.run(["pip", "install", "PyPDF2"], check=False)
    import PyPDF2
    print("PyPDF2 installed successfully!")

chat_model = None
stop_generation = False
uploaded_pdf_content = ""

# Path to local GGUF model — set MODEL_PATH env var or edit this default
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "foundation-sec-8b-instruct-q8_0.gguf")

def load_local_gguf_model():
    """Load local GGUF model — optimized for RTX 4060 Laptop"""
    global chat_model

    if not os.path.exists(LOCAL_MODEL_PATH):
        return f"Model file not found: {LOCAL_MODEL_PATH}"

    try:
        print(f"Loading local GGUF: {LOCAL_MODEL_PATH}")

        gc.collect()

        # RTX 4060 Laptop optimized settings
        chat_model = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=2048,
            n_gpu_layers=-1,     # full GPU offload
            n_batch=1024,
            n_ubatch=256,
            n_threads=8,
            n_predict=-1,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mmap=True,
            use_mlock=True,
            verbose=False,
            main_gpu=0,
            tensor_split=None,
            low_vram=True,       # required for laptop GPU
            mul_mat_q=True,
            offload_kqv=True,
            flash_attn=False,
            numa=False,
        )

        return "Model loaded — long-text mode (600-700 tokens)"
    except Exception as e:
        return f"Load failed: {str(e)}"

def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF"""
    global uploaded_pdf_content

    if pdf_file is None:
        uploaded_pdf_content = ""
        return "No file uploaded"

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_content = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text() + "\n\n"

        uploaded_pdf_content = text_content.strip()

        if uploaded_pdf_content:
            return f"PDF loaded: {len(pdf_reader.pages)} pages, {len(uploaded_pdf_content)} chars"
        else:
            return "PDF loaded but no text could be extracted"

    except Exception as e:
        uploaded_pdf_content = ""
        return f"PDF error: {str(e)}"

def clear_pdf():
    global uploaded_pdf_content
    uploaded_pdf_content = ""
    return "PDF cleared", None

def stop_generation_fn():
    global stop_generation
    stop_generation = True
    return "Generation stopped"

def truncate_long_text(text, max_chars=1000):
    if len(text) <= max_chars:
        return text
    front = text[:max_chars//2]
    back = text[-(max_chars//2):]
    return f"{front}\n\n[...truncated...]\n\n{back}"

def local_gguf_stream(message, history, system_prompt, temperature, max_tokens, speed_display=None):
    """Streaming inference with local GGUF model"""
    global chat_model, stop_generation, uploaded_pdf_content

    stop_generation = False

    if chat_model is None:
        history.append([message, "Please load the model first!"])
        yield history, "", "Load model first"
        return

    if not message.strip():
        yield history, "", "Waiting for input..."
        return

    full_message = message
    if uploaded_pdf_content:
        full_message = f"Answer based on the following PDF content:\n\n{uploaded_pdf_content}\n\nQuestion: {message}"

    original_length = len(full_message)
    if original_length > 2000:
        full_message = truncate_long_text(full_message, 1800)
        print(f"\nTruncated: {original_length} -> {len(full_message)} chars")

    if system_prompt.strip():
        prompt = f"{system_prompt.strip()}\n\n"
    else:
        prompt = "You are a professional AI security assistant. Answer questions clearly and accurately.\n\n"
    prompt += f"Human: {full_message}\nAssistant:"

    try:
        init_start = time.time()
        start_time = time.time()
        token_count = 0

        dynamic_max_tokens = max_tokens
        print(f"Max tokens: {dynamic_max_tokens}")
        print(f"\nStarting streaming generation...")

        stream = chat_model.create_completion(
            prompt,
            max_tokens=dynamic_max_tokens,
            temperature=0.3,
            top_p=0.5,
            top_k=10,
            repeat_penalty=1.0,
            stop=[],
            echo=False,
            stream=True,
            seed=42,
        )

        response_text = ""
        last_update = start_time
        first_token_time = None

        for chunk in stream:
            try:
                if stop_generation:
                    print("\nUser stopped generation")
                    response_text += " [stopped]"
                    break

                if first_token_time is None:
                    first_token_time = time.time()
                    delay = first_token_time - init_start
                    print(f"\nFirst token: {delay:.2f}s")

                token = ""
                if chunk and isinstance(chunk, dict):
                    choices = chunk.get('choices', [])
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        choice = choices[0]
                        if isinstance(choice, dict):
                            token = choice.get('text', '')

                if token and isinstance(token, str) and len(token) > 0:
                    token_count += 1
                    response_text += token

                    current_time = time.time()
                    if current_time - last_update >= 0.2:
                        elapsed = current_time - start_time
                        speed = token_count / elapsed if elapsed > 0 else 0

                        current_history = history + [[message, response_text]]
                        input_info = f"[{len(message)} chars]" if len(message) > 100 else ""
                        pdf_info = " [+PDF]" if uploaded_pdf_content else ""
                        speed_text = f"⚡ {speed:.2f} tok/s | {token_count} tokens {input_info}{pdf_info}"

                        yield current_history, "", speed_text
                        last_update = current_time

            except Exception as chunk_error:
                print(f"Chunk error (continuing): {chunk_error}")
                continue

        current_history = history + [[message, response_text]]

        if len(current_history) > 10:
            current_history = current_history[-5:]

        final_time = time.time() - start_time
        final_speed = token_count / final_time if final_time > 0 else 0
        input_info = f"| input: {original_length} chars" if original_length > 100 else ""
        pdf_info = " | +PDF" if uploaded_pdf_content else ""
        final_speed_text = f"Done: {final_speed:.2f} tok/s | {token_count} tokens {input_info}{pdf_info}"

        yield current_history, "", final_speed_text

        print(f"\nFinal: {final_speed:.2f} tok/sec | Tokens: {token_count} | Time: {final_time:.1f}s")

    except Exception as e:
        print(f"Stream error: {e}")
        error_response = f"Streaming error: {str(e)}"
        yield history + [[message, error_response]], "", f"Error: {str(e)}"

LOCAL_SYSTEM = """You are a professional AI security assistant specializing in vulnerability analysis and remediation. Provide clear, accurate, and actionable security guidance."""

with gr.Blocks(title="Foundation-Sec-8B Local GGUF + PDF", theme=gr.themes.Default()) as demo:
    gr.Markdown("# Foundation-Sec-8B + PDF Analysis")
    gr.Markdown("**Local GGUF | GPU Accelerated | PDF Document Analysis | Live Speed Display**")
    gr.Markdown(f"**Model:** `{os.path.basename(LOCAL_MODEL_PATH)}`")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450, show_label=False)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Enter message (supports PDF content analysis — target 7+ tok/s)",
                    show_label=False,
                    container=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Local GGUF Model")
            model_info = gr.Textbox(
                label="Model file",
                value=os.path.basename(LOCAL_MODEL_PATH),
                interactive=False
            )
            load_btn = gr.Button("Load GGUF Model", variant="secondary")
            status = gr.Textbox(
                label="Status",
                value="Not loaded",
                interactive=False
            )

            gr.Markdown("### PDF Upload")
            pdf_file = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )
            pdf_status = gr.Textbox(
                label="PDF Status",
                value="No file",
                interactive=False
            )
            clear_pdf_btn = gr.Button("Clear PDF", variant="secondary")

            gr.Markdown("### Settings")
            system_prompt = gr.Textbox(
                label="System prompt",
                value=LOCAL_SYSTEM,
                lines=3
            )

            gr.Markdown("### Generation Parameters")
            temperature = gr.Slider(0.1, 1.5, value=0.1, label="Temperature")
            max_tokens = gr.Slider(500, 700, value=600, label="Max Tokens")

            gr.Markdown("### Live Stats")
            speed_display = gr.Textbox(
                label="Speed",
                value="Waiting...",
                interactive=False,
                lines=1
            )

    load_btn.click(load_local_gguf_model, outputs=[status])

    pdf_file.change(extract_pdf_text, inputs=[pdf_file], outputs=[pdf_status])

    clear_pdf_btn.click(clear_pdf, outputs=[pdf_status, pdf_file])

    send_btn.click(
        local_gguf_stream,
        inputs=[msg, chatbot, system_prompt, temperature, max_tokens, speed_display],
        outputs=[chatbot, msg, speed_display]
    )

    msg.submit(
        local_gguf_stream,
        inputs=[msg, chatbot, system_prompt, temperature, max_tokens, speed_display],
        outputs=[chatbot, msg, speed_display]
    )

    stop_btn.click(stop_generation_fn, outputs=[speed_display])

    clear_btn.click(lambda: ([], "Waiting..."), outputs=[chatbot, speed_display])

if __name__ == "__main__":
    print("Foundation-Sec-8B + PDF starting...")
    print(f"Model: {LOCAL_MODEL_PATH}")
    print("Set MODEL_PATH env var to point to your GGUF file")
    print("Opening http://localhost:8071")

    demo.launch(
        server_name="127.0.0.1",
        server_port=8071,
        share=False,
        inbrowser=True,
        debug=False,
        quiet=False
    )
