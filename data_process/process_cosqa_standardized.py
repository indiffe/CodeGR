import json
import argparse
from pathlib import Path
from tqdm import tqdm


def process_cosqa(
    input_path: str,
    output_path: str,
    start_id: int = 0,
    prefix: str = "cosqa",
):
    """
    CoSQA → DSI / CodeGR 标准化数据

    规则：
    - 只保留 label == 1
    - query = data["doc"]
    - code  = data["code"]
    - docid = f"{prefix}_{global_id:08d}"
    - train / dev / test 共用同一 docid 命名空间（通过 start_id 控制）
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert input_path.exists(), f"Input file not found: {input_path}"

    with input_path.open("r", encoding="utf-8") as f:
        data_list = json.load(f)

    kept = 0
    skipped = 0
    global_id = start_id

    with output_path.open("w", encoding="utf-8") as fout:
        for data in tqdm(data_list, desc=f"Processing {input_path.name}"):

            if data.get("label", 0) != 1:
                skipped += 1
                continue

            query = data["doc"].strip()
            code = data["code"].strip()

            docid = f"{prefix}_{global_id:08d}"
            global_id += 1

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

    print("========== CoSQA → DSI STANDARDIZED DONE ==========")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Kept   : {kept}")
    print(f"Skipped: {skipped}")
    print(f"LastID : {global_id - 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CoSQA for DSI / CodeGR with standardized docids"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="Global start id to ensure train/dev/test share one ID space",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="cosqa",
        help="DocID prefix (default: cosqa)",
    )

    args = parser.parse_args()

    process_cosqa(
        input_path=args.input,
        output_path=args.output,
        start_id=args.start_id,
        prefix=args.prefix,
    )
