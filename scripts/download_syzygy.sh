#!/bin/bash
# Download Syzygy 3-4-5 piece tablebases (~1GB) to the syzygy/ directory.
# Safe to re-run — curl -C - resumes partial files and skips complete ones.
set -e
DEST="${1:-E:/dev/chess-ai/syzygy}"
BASE="http://tablebase.sesse.net/syzygy/3-4-5"
mkdir -p "$DEST"
cd "$DEST"

echo "Fetching file list from $BASE..."
FILES=$(curl -s "$BASE/" | grep -oE 'href="[^"]+\.rtb[wz]"' | sed 's/href="//;s/"$//')
COUNT=$(echo "$FILES" | wc -l)
echo "Found $COUNT files. Downloading to $DEST..."

i=0
for f in $FILES; do
    i=$((i + 1))
    # Skip if already complete (size > 0 and no .tmp suffix would suffice but curl -C - handles it)
    if [ -f "$f" ]; then
        printf "[%3d/%d] %s (exists, skipping)\n" "$i" "$COUNT" "$f"
        continue
    fi
    printf "[%3d/%d] %s\n" "$i" "$COUNT" "$f"
    curl -sS -C - -o "$f" "$BASE/$f"
done

echo "Done. Total size:"
du -sh "$DEST"
