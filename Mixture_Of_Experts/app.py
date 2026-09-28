"""Web UI: generate text, see which experts handled each character, and review training.

    python app.py                      # http://127.0.0.1:7860
    python app.py --share              # temporary public link
    python app.py --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
import os
from itertools import groupby

import gradio as gr
import torch
from moe.data import CharTokenizer
from moe import load_model_from_checkpoint
from plot_metrics import curves_figure, load_figure, read_metrics

EXPERT_COLORS = ["#6C8CD5", "#E3A33B", "#5BB08F", "#D46A6A",
                 "#9A7BC9", "#4FB3C4", "#C98A5B", "#8C9A5B"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
.gradio-container { max-width: 1100px !important; margin: auto; }
#title h1 { font-size: 2rem; margin-bottom: 0.2rem; }
#title p { color: #5A6275; margin-top: 0; }
#output textarea { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.05rem; line-height: 1.6; }
#routing { font-family: 'Source Serif 4', Georgia, serif; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
)


class Demo:
    def __init__(self, ckpt_path: str, metrics_path: str):
        self.metrics_path = metrics_path
        self.error = None
        if not os.path.exists(ckpt_path):
            self.error = f"No checkpoint at `{ckpt_path}`. Train one with `python train.py`."
            return
        self.model, ckpt = load_model_from_checkpoint(ckpt_path)
        self.tok = CharTokenizer.from_dict(ckpt["tokenizer"])
        self.device = next(self.model.parameters()).device
        self.ckpt = ckpt

    # ---- generation -------------------------------------------------------
    def generate(self, prompt, length, temperature, top_k, layer, seed):
        if self.error:
            raise gr.Error(self.error)
        if seed >= 0:
            torch.manual_seed(int(seed))
        ids = self.tok.encode(prompt) or self.tok.encode("\n")
        x = torch.tensor([ids], device=self.device)
        out = self.model.generate(x, int(length), temperature=float(temperature),
                                  top_k=int(top_k) if top_k > 0 else None)
        text = self.tok.decode(out[0].tolist())

        # One more pass over the final text to record routing decisions.
        window = out[:, -self.model.cfg.max_seq_len:]
        with torch.no_grad():
            routing = self.model(window, return_routing=True).routing
        self._last = (self.tok.decode(window[0].tolist()), [r[0].cpu() for r in routing])
        return text, *self.routing_views(layer)

    def routing_views(self, layer):
        if not getattr(self, "_last", None):
            return [], None
        text, routing = self._last
        layer = int(layer)
        top1 = routing[layer][:, 0].tolist()
        # Merge runs of characters that went to the same expert.
        spans, pos = [], 0
        for expert, run in groupby(top1):
            n = len(list(run))
            spans.append((text[pos:pos + n], f"Expert {expert}"))
            pos += n
        return spans, self.usage_figure(routing)

    def usage_figure(self, routing):
        e = self.model.cfg.num_experts
        load = []
        for r in routing:
            c = torch.bincount(r.flatten(), minlength=e).float()
            load.append((c / c.sum()).tolist())
        fig = load_figure([{"expert_load": load}])
        fig.axes[0].set_title("Share of tokens per expert, this text")
        return fig

    # ---- training tab -----------------------------------------------------
    def training_views(self):
        if not os.path.exists(self.metrics_path):
            return "No metrics yet. Run `python train.py` to create them.", None, None
        rows = read_metrics(self.metrics_path)
        best = min(rows, key=lambda r: r["val_loss"])
        n = self.model.num_params() if not self.error else None
        cfg = self.model.cfg if not self.error else None
        md = (
            f"**Best validation loss** {best['val_loss']:.3f} at step {best['step']}  \n"
            f"**Validation accuracy** {best['val_acc']:.1%} (next character)  \n"
            f"**Training time** {rows[-1]['elapsed_s'] / 60:.1f} min for {rows[-1]['step']} steps"
        )
        if n:
            md += (f"  \n**Model** {n['total'] / 1e6:.2f}M parameters, "
                   f"{n['active_per_token'] / 1e6:.2f}M active per token "
                   f"({cfg.num_layers} layers, {cfg.num_experts} experts, top-{cfg.top_k})")
        curves = curves_figure(rows)
        load = load_figure(rows)
        return md, curves, load


def build_ui(demo: Demo) -> gr.Blocks:
    n_layers = demo.model.cfg.num_layers if not demo.error else 1
    n_experts = demo.model.cfg.num_experts if not demo.error else len(EXPERT_COLORS)
    window = demo.model.cfg.max_seq_len if not demo.error else 0
    color_map = {f"Expert {i}": EXPERT_COLORS[i % len(EXPERT_COLORS)] for i in range(n_experts)}

    kwargs = {} if int(gr.__version__.split(".")[0]) >= 6 else {"theme": THEME, "css": CSS}
    with gr.Blocks(title="MoE Transformer", **kwargs) as ui:
        with gr.Column(elem_id="title"):
            gr.Markdown(
                "# Mixture-of-Experts Transformer\n"
                "A small language model where every layer sends each character to 2 of "
                f"{n_experts} expert networks. Write a prompt and watch where it goes."
            )
        if demo.error:
            gr.Markdown(f"> {demo.error}")

        with gr.Tab("Try it"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(label="Prompt", value="ROMEO:\n", lines=3)
                    output = gr.Textbox(label="Generated text", lines=12, elem_id="output")
                with gr.Column(scale=2):
                    length = gr.Slider(20, 1000, value=300, step=10, label="Characters to generate")
                    temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature",
                                            info="0 always picks the likeliest character")
                    top_k = gr.Slider(0, 65, value=20, step=1, label="Top-k",
                                      info="Sample only from the k likeliest characters (0 = off)")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    run = gr.Button("Generate", variant="primary")

            gr.Markdown("### Which expert handled each character")
            layer = gr.Radio(list(range(n_layers)), value=n_layers - 1, label="Layer")
            routing = gr.HighlightedText(label=f"Top-1 expert per character (last {window} characters)", color_map=color_map,
                                         combine_adjacent=False, show_legend=True, elem_id="routing")
            usage = gr.Plot(label="Expert load")

            run.click(demo.generate, [prompt, length, temperature, top_k, layer, seed],
                      [output, routing, usage])
            prompt.submit(demo.generate, [prompt, length, temperature, top_k, layer, seed],
                          [output, routing, usage])
            layer.change(demo.routing_views, [layer], [routing, usage])

            gr.Examples([["ROMEO:\n"], ["JULIET:\nO "], ["KING HENRY:\nWhat say you"], ["To be, or not"]],
                        inputs=[prompt])

        with gr.Tab("Training"):
            summary = gr.Markdown()
            curves = gr.Plot(label="Loss and accuracy")
            load = gr.Plot(label="Expert load on validation data")
            refresh = gr.Button("Refresh")
            refresh.click(demo.training_views, None, [summary, curves, load])
            ui.load(demo.training_views, None, [summary, curves, load])

    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/model.pt")
    ap.add_argument("--metrics", default="checkpoints/metrics.jsonl")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()

    ui = build_ui(Demo(a.checkpoint, a.metrics))
    launch = {"server_name": a.host, "server_port": a.port, "share": a.share}
    if int(gr.__version__.split(".")[0]) >= 6:
        launch.update(theme=THEME, css=CSS)
    ui.launch(**launch)


if __name__ == "__main__":
    main()
