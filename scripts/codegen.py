"""
Generate code for trained models. Python, Go & Typescript
"""

import json
import pathlib

from lab.models import Model, MODELS

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

def do_python_model(model: Model, data: dict) -> str:
    pass

def get_model(path: pathlib.Path) -> Model | None:
    for model in MODELS:
        label = path.stem[6:] if path.stem.startswith("model-") else path.stem
        if model.label == label:
            return model
    return None

def main():
    code = []
    for file in pathlib.Path("output").glob("*.json"):
        model = get_model(file)
        if model is None:
            raise ValueError(f"Unknown model: {file}")

        emo_model_dict = json.loads(file.read_text())
        code.append(do_go_model(model, emo_model_dict))
        print(f"{model.label}: {emo_model_dict is not None}")
    pathlib.Path("go/emopoint/generated.go").write_text("package emopoint\n\n" + ("\n\n".join(code)))

if __name__ == "__main__":
    main()