#!/usr/bin/env python3
"""Sem3D test script: run tokenizer on example text + TinyStories."""

from sem3d_tokenizer import Sem3DTokenizer

def run_test(tokenizer, name, text):
    """Run encode/decode on text, verify lossless, return formatted output."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"TEST: {name}")
    lines.append(f"{'='*60}")
    lines.append(f"\nInput ({len(text)} chars):")
    lines.append(text[:200] + ("..." if len(text) > 200 else ""))
    lines.append("")

    tag_dict, stream = tokenizer.encode(text)

    lines.append("--- 2D Dictionary ---")
    lines.append(tokenizer.format_dictionary(tag_dict))
    lines.append(f"\nTotal tags: {len(tag_dict)}")
    lines.append("")

    lines.append("--- Coordinate Stream ---")
    formatted = tokenizer.format_stream(stream)
    lines.append(formatted[:2000] + ("..." if len(formatted) > 2000 else ""))
    lines.append(f"\nTotal coordinates: {len(stream)}")
    lines.append("")

    decoded = tokenizer.decode(tag_dict, stream)
    lossless = decoded == text

    lines.append(f"--- Lossless Roundtrip: {lossless} ---")
    if not lossless:
        # Find first difference
        for i, (a, b) in enumerate(zip(text, decoded)):
            if a != b:
                ctx = 20
                lines.append(f"First diff at char {i}:")
                lines.append(f"  Original: ...{repr(text[max(0,i-ctx):i+ctx])}...")
                lines.append(f"  Decoded:  ...{repr(decoded[max(0,i-ctx):i+ctx])}...")
                break
        if len(text) != len(decoded):
            lines.append(f"  Length: original={len(text)}, decoded={len(decoded)}")
    lines.append("")

    # Stream statistics
    tag_types = {}
    for coord in stream:
        tag_types[coord.tag] = tag_types.get(coord.tag, 0) + 1
    lines.append("--- Tag Distribution ---")
    for tag, count in sorted(tag_types.items(), key=lambda x: -x[1]):
        lines.append(f"  {tag:<10} {count:>4}")
    lines.append("")

    return "\n".join(lines), lossless


def main():
    tokenizer = Sem3DTokenizer()
    results = []
    all_pass = True

    # Test 1: Example from RESEARCHPHASE12.md
    example = ("Ben was walking through the store when he came across a very "
               "special vase. When Ben saw it he was amazed! He said, 'Wow, "
               "that is a really amazing vase!'")
    output, ok = run_test(tokenizer, "Example Text", example)
    results.append(output)
    if not ok:
        all_pass = False

    # Test 2: 1000 chars from TinyStories middle
    try:
        with open("data/tiny_stories_100m") as f:
            f.seek(50_000_000)
            # Read to next sentence boundary for clean start
            f.readline()
            chunk = f.read(1000)
            # Trim to last sentence boundary
            last_period = chunk.rfind(".")
            if last_period > 0:
                chunk = chunk[:last_period + 1]
        output, ok = run_test(tokenizer, "TinyStories (1000 chars from middle)", chunk)
        results.append(output)
        if not ok:
            all_pass = False
    except FileNotFoundError:
        results.append("SKIPPED: data/tiny_stories_100m not found")

    # Summary
    results.append("=" * 60)
    results.append(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    results.append("=" * 60)

    full_output = "\n".join(results)
    print(full_output)

    with open("phase12_results.txt", "w") as f:
        f.write(full_output + "\n")
    print(f"\nResults saved to phase12_results.txt")


if __name__ == "__main__":
    main()
