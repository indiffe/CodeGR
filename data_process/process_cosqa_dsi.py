import json
import argparse
from pathlib import Path
from tqdm import tqdm


def process_cosqa(input_path: str, output_path: str):
    """
    CoSQA → DSI 预训练数据
    - 原始数据是 JSON array（不是 jsonl）
    - 仅保留 label == 1
    - 使用 doc 作为 NL query
    - 输出 jsonl: (query, code, docid)
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert input_path.exists(), f"Input file not found: {input_path}"

    with input_path.open("r", encoding="utf-8") as f:
        data_list = json.load(f)

    kept = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for data in tqdm(data_list, desc="Processing CoSQA for DSI"):

            if data.get("label", 0) != 1:
                skipped += 1
                continue

            query = data["doc"].strip()
            code = data["code"].strip()
            docid = data["idx"]

            fout.write(
                json.dumps(
                    {
                        "query": query,
                        "code": code,
                        "docid": docid,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    print("========== CoSQA → DSI DONE ==========")
    print(f"Kept   : {kept}")
    print(f"Skipped: {skipped}")
    print(f"Saved  : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CoSQA for DSI pretraining")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to original CoSQA json file (JSON array)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output DSI jsonl file",
    )

    args = parser.parse_args()
    process_cosqa(args.input, args.output)
