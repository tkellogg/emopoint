"""
Generate code for trained models. Python, Go & Typescript
"""

import json
import pathlib

from lab.models import Model, MODELS

###############################################
### Language: Go
###############################################

def do_go_model(model: Model, data: dict) -> str:
    def format_dim(dim: dict) -> str:
        return f"""{{Negative: "{dim["negative"]}", Positive: "{dim["positive"]}"}},"""

    def format_arr(weight_arr) -> str:
        if isinstance(weight_arr[0], list):
            weight_arr = weight_arr[0] 
        buf = ["\n\t\t{"]
        for i, w in enumerate(weight_arr):
            assert isinstance(w, float), type(w)
            if i % 4 == 0:
                buf.append("\n\t\t\t")
            buf.append(f"{w},")
        buf.append("\n\t\t},")
        return "".join(buf)

    dims = "\n\t\t".join(format_dim(d) for d in data["dims"])
    weight_arr = data["weights"] if len(data["weights"]) == 3 else data["weights"][0]
    weights = "\n\t\t".join(format_arr(w) for w in weight_arr)
    proper_name = (
        model.label
        .upper()
        .replace("(", "").replace(")", "")
        .replace("-", "_")
    )
    code = f"""var {proper_name} = EmoModel{{
	Dims: []DimLabel{{
		{dims}
	}},
	numEmbeddingDimensions: {data["num_emb_dims"]},
	Weights: [][]float32{{{weights}
	}},
}}
"""
    if model.docs:
        return f"// {model.docs}\n{code}"
    else:
        return code


###############################################
### Language: Python
###############################################

def do_python_model(model: Model, data: dict) -> str:
    def format_dim(dim: dict) -> str:
        return f"""DimLabel(negative="{dim["negative"]}", positive="{dim["positive"]}"),"""

    def format_arr(weight_arr) -> str:
        if isinstance(weight_arr[0], list):
            weight_arr = weight_arr[0] 
        buf = ["\n        ["]
        for i, w in enumerate(weight_arr):
            assert isinstance(w, float), type(w)
            if i % 4 == 0:
                buf.append("\n            ")
            buf.append(f"{w},")
        buf.append("\n        ],")
        return "".join(buf)

    dims = "\n        ".join(format_dim(d) for d in data["dims"])
    weight_arr = data["weights"] if len(data["weights"]) == 3 else data["weights"][0]
    weights = "\n        ".join(format_arr(w) for w in weight_arr)
    proper_name = (
        model.label
        .lower()
        .replace("(", "").replace(")", "")
        .replace("-", "_")
    )
    code = f"""
{proper_name} = EmoModel(
	dims=[
		{dims}
	],
	num_emb_dims={data["num_emb_dims"]},
	weights=np.array([{weights}
	]),
)
"""
    if model.docs:
        return f'{code}"""{model.docs}"""'
    else:
        return code


###############################################
### Language: Typescript
###############################################

def do_ts_model(model: Model, data: dict) -> str:
    def format_dim(dim: dict) -> str:
        return f"""new DimLabel("{dim["negative"]}", "{dim["positive"]}"),"""

    def format_arr(weight_arr) -> str:
        if isinstance(weight_arr[0], list):
            weight_arr = weight_arr[0] 
        buf = ["\n    new Float32Array(["]
        for i, w in enumerate(weight_arr):
            assert isinstance(w, float), type(w)
            if i % 4 == 0:
                buf.append("\n      ")
            buf.append(f"{w},")
        buf.append("\n    ]),")
        return "".join(buf)

    dims = "\n    ".join(format_dim(d) for d in data["dims"])
    weight_arr = data["weights"] if len(data["weights"]) == 3 else data["weights"][0]
    weights = "\n    ".join(format_arr(w) for w in weight_arr)
    proper_name = (
        model.label
        .upper()
        .replace("(", "").replace(")", "")
        .replace("-", "_")
    )
    code = f"""
const {proper_name} = EmoModel([{weights}
  ],
  [
    {dims}
  ],
  {data["num_emb_dims"]},
)
"""
    if model.docs:
        return f'/** \n * {model.docs}\n */\n{code}'
    else:
        return code


###############################################
### Main
###############################################

def get_model(path: pathlib.Path) -> Model | None:
    for model in MODELS:
        label = path.stem[6:] if path.stem.startswith("model-") else path.stem
        if model.label == label:
            return model
    return None

def main():
    go_code = []
    python_code = []
    ts_code = []
    for file in pathlib.Path("output").glob("*.json"):
        model = get_model(file)
        if model is None:
            raise ValueError(f"Unknown model: {file}")

        emo_model_dict = json.loads(file.read_text())
        go_code.append(do_go_model(model, emo_model_dict))
        python_code.append(do_python_model(model, emo_model_dict))
        ts_code.append(do_ts_model(model, emo_model_dict))
        print(f"{model.label}: {emo_model_dict is not None}")
    pathlib.Path("go/emopoint/generated.go").write_text("package emopoint\n\n" + ("\n\n".join(go_code)))
    pathlib.Path("emopoint/generated.py").write_text("import numpy as np\n\nfrom emopoint.lib import EmoModel, DimLabel\n\n" + ("\n\n".join(python_code)))
    pathlib.Path("js/src/generated.ts").write_text("import {EmoModel, DimLabel} from .\n\n" + ("\n\n".join(ts_code)))

if __name__ == "__main__":
    main()