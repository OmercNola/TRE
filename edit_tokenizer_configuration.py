import json

path_for_tokenizer = Path().resolve().parent / 'models/tokenizer/'

tokenizer_config_path = Path(f"{path_for_tokenizer}/vocab.json")
with open(tokenizer_config_path) as f:
    tokenizer_json = json.load(f)

my_token = "<e1>"
max_identifier = max(tokenizer_json["model"]["vocab"].values())
tokenizer_json["model"]["vocab"][my_token] = max_identifier + 1

with open(tokenizer_config_path.replace(".json", ".new.json"), "w") as f:
    tokenizer_json = json.dump(tokenizer_json, f)

from transformers import AutoTokenizer

from preprocessing import undot


class UndotStrategy:
    REPLACE = "replace"
    EXTEND = "extend"


def undot_vocab_file(source_tokenizer_dir, target_dir, strategy=UndotStrategy.REPLACE):
    with open(os.path.join(source_tokenizer_dir, "vocab.txt")) as vocab_f:
        original_lines = [line for line in vocab_f]

    if strategy == UndotStrategy.EXTEND:
        extra_lines = [undotted for line in original_lines if (undotted := undot(line)) not in original_lines]
        vocab_lines = original_lines + extra_lines
    else:
        vocab_lines = [undot(line) for line in original_lines]

    with open(os.path.join(target_dir, "vocab.txt", "w")) as f:
        f.writelines(vocab_lines)


def undot_tokenizer_json(tokenizer_folder, strategy=UndotStrategy.REPLACE):
    with open(tokenizer_folder) as f:
        tokenizer_json = json.load(f)

    new_vocab = {}
    if strategy == UndotStrategy.EXTEND:
        new_vocab.update(tokenizer_json["model"]["vocab"])

    for key, value in tokenizer_json["model"]["vocab"].items():
        new_vocab[key] = value
        undotted = undot(key)
        if undotted not in new_vocab:
            new_vocab[undotted] = value

    tokenizer_json["model"]["vocab"] = new_vocab

    with open(os.path.join(tokenizer_folder, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_json, f)


def undot_tokenizer_vocab(source_tokenizer_dir, target_dir, strategy=UndotStrategy.REPLACE):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(target_dir)

    undot_vocab_file(source_tokenizer_dir, target_dir, strategy)
    undot_tokenizer_json(target_dir, strategy)