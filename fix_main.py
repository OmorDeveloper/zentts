import sys
import tokenize
import io

def fix_file(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    source = source.replace(
        "from ZENTTS_onnx import ZENTTS",
        "from kokoro_onnx import Kokoro",
    )

    all_tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    result = []
    for idx, tok in enumerate(all_tokens):
        tok_type, tok_string, start, end, line = tok
        if tok_type == tokenize.NAME and tok_string == "ZENTTS":
            next_meaningful = None
            for nxt in all_tokens[idx + 1:]:
                if nxt.type not in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
                    next_meaningful = nxt
                    break
            if next_meaningful is not None and next_meaningful.string == "(":
                tok_string = "Kokoro"
            else:
                tok_string = "engine"
        result.append((tok_type, tok_string))

    fixed = tokenize.untokenize(result)

    out_path = path.replace("__main__.py", "__main__.fixed.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(fixed)
    print(f"Fixed file written to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_main.py path/to/__main__.py")
        sys.exit(1)
    fix_file(sys.argv[1])