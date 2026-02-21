#!/usr/bin/env python3
"""Benchmark Sem3D tokenizer on full tiny_stories_100m."""

import time
import sys
from sem3d_tokenizer import TagQualityTokenizer

CHUNK_SIZE = 10_000  # chars per chunk — split on sentence boundaries
DATA_PATH = "data/tiny_stories_100m"


def main():
    tokenizer = TagQualityTokenizer()

    total_chars = 0
    total_coords = 0
    total_dict_entries = 0
    total_chunks = 0
    lossless_failures = 0
    quality_counts = {}
    tag_type_counts = {}

    file_size = 100_000_000
    start = time.time()

    with open(DATA_PATH) as f:
        buf = ""
        bytes_read = 0

        while True:
            raw = f.read(CHUNK_SIZE)
            if not raw:
                # Process remaining buffer
                if buf.strip():
                    tag_dict, stream = tokenizer.encode(buf)
                    decoded = tokenizer.decode(tag_dict, stream)
                    if decoded != buf:
                        lossless_failures += 1
                    total_chars += len(buf)
                    total_coords += len(stream)
                    total_dict_entries += len(tag_dict)
                    total_chunks += 1
                    for c in stream:
                        tag_type_counts[c.tag] = tag_type_counts.get(c.tag, 0) + 1
                        if c.tag not in ('PREP', 'CONJ', 'PUNCT', 'ADV', 'DET',
                                         'AUX', 'WORD', 'INTJ', 'PRON', 'ADJ',
                                         'PROPN', 'NUM', 'PART', 'WS'):
                            quality_counts[c.quality] = quality_counts.get(c.quality, 0) + 1
                break

            buf += raw
            bytes_read += len(raw.encode('utf-8'))

            # Split on last sentence boundary
            last_period = buf.rfind(". ")
            if last_period < 0:
                last_period = buf.rfind(".\n")
            if last_period < 0 and len(buf) > CHUNK_SIZE * 2:
                # Force split at chunk boundary
                last_period = CHUNK_SIZE

            if last_period > 0:
                chunk = buf[:last_period + 1]
                buf = buf[last_period + 1:]

                tag_dict, stream = tokenizer.encode(chunk)
                decoded = tokenizer.decode(tag_dict, stream)
                if decoded != chunk:
                    lossless_failures += 1

                total_chars += len(chunk)
                total_coords += len(stream)
                total_dict_entries += len(tag_dict)
                total_chunks += 1

                for c in stream:
                    tag_type_counts[c.tag] = tag_type_counts.get(c.tag, 0) + 1
                    if c.tag not in ('PREP', 'CONJ', 'PUNCT', 'ADV', 'DET',
                                     'AUX', 'WORD', 'INTJ', 'PRON', 'ADJ',
                                     'PROPN', 'NUM', 'PART', 'WS'):
                        quality_counts[c.quality] = quality_counts.get(c.quality, 0) + 1

                # Progress
                elapsed = time.time() - start
                rate = total_chars / elapsed if elapsed > 0 else 0
                eta = (file_size - total_chars) / rate if rate > 0 else 0
                pct = total_chars / file_size * 100
                ratio = total_chars / total_coords if total_coords > 0 else 0
                print(f"\r{pct:5.1f}% | {total_chars/1e6:.1f}MB | "
                      f"{total_coords:,} coords | {ratio:.2f} chars/coord | "
                      f"{rate/1e3:.1f} KB/s | ETA {eta:.0f}s | "
                      f"failures={lossless_failures}",
                      end="", flush=True)

    elapsed = time.time() - start
    print()
    print()
    print("=" * 60)
    print("RESULTS: Phase 12 Tag-Quality Tokenizer on tiny_stories_100m")
    print("=" * 60)
    print(f"Total chars:           {total_chars:>15,}")
    print(f"Total coordinates:     {total_coords:>15,}")
    print(f"Chars/coordinate:      {total_chars/total_coords:>15.2f}")
    print(f"Total chunks:          {total_chunks:>15,}")
    print(f"Avg dict entries/chunk:{total_dict_entries/total_chunks:>15.1f}")
    print(f"Lossless failures:     {lossless_failures:>15,}")
    print(f"Time:                  {elapsed:>15.1f}s")
    print(f"Rate:                  {total_chars/elapsed/1e3:>15.1f} KB/s")
    print()
    print("--- Semantic Quality Distribution ---")
    for q, count in sorted(quality_counts.items(), key=lambda x: -x[1]):
        print(f"  {q:<20} {count:>10,}")
    print()
    print("--- Tag Type Distribution ---")
    for t, count in sorted(tag_type_counts.items(), key=lambda x: -x[1]):
        pct = count / total_coords * 100
        print(f"  {t:<10} {count:>12,}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
