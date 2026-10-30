"""Point d'entrée Hugging Face Space (SDK Gradio, matériel ZeroGPU gratuit).

Le SDK Docker étant payant, on sert l'API de service.py depuis une Space
Gradio : c'est Gradio qui lance le serveur (ZeroGPU ne démarre qu'à travers
demo.launch()), et les routes /ai/* y sont greffées, identiques pour le
backend Go.
"""

import spaces  # avant tout paquet qui touche à CUDA (exigence ZeroGPU)

import threading

import gradio as gr

import service


# ZeroGPU refuse de démarrer sans au moins une fonction @spaces.GPU.
# Le service tourne entièrement sur CPU : celle-ci n'est jamais appelée.
@spaces.GPU
def _unused_gpu_hook():
    return None


with gr.Blocks(title="Yélé AI") as demo:
    gr.Markdown("# Yélé AI\nAPI : `/ai/health`, `/ai/anomalies`, "
                "`/ai/forecast`, `/ai/retrain`.")

# ssr_mode=False : sur les Spaces, Gradio place sinon un serveur Node devant
# Python, qui sert la page Gradio et ne relaie pas les routes /ai/*.
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False,
            prevent_thread_lock=True)

# Routes FastAPI de service.py ajoutées en tête du serveur de Gradio.
ai_routes = [r for r in service.app.routes
             if getattr(r, "path", "").startswith("/ai/")]
demo.app.router.routes[:0] = ai_routes

# Le lifespan de service.py ne s'exécute pas ici : entraînement initial lancé
# à la main, en arrière-plan.
threading.Thread(target=service._retrain, daemon=True).start()

demo.block_thread()
