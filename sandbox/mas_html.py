import sys
from pathlib import Path

from transformer_lens import HookedTransformer  # type: ignore[import]

from thesis.device import Device
from thesis.mas import WeightedSamplesStore, html

model_name = sys.argv[1]
mas_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
indices_string = sys.argv[4]
if ":" in indices_string:
    [start, end] = indices_string.split(":")
    indices = list(range(int(start), int(end)))
else:
    indices = [int(index) for index in indices_string.split(",")]

device = Device.get()
model = HookedTransformer.from_pretrained(model_name, device=device.torch())
store = WeightedSamplesStore.load(mas_path, device)

for index in indices:
    html_str = html.generate_html(model, store.feature_samples()[index], store.feature_activations()[index])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{index}.html", "w") as f:
        f.write(html_str)
