OUT=/mnt/swx/ThinkVLN/results/watcher_rollout_train_streamvln_rgb_v2_out
DST=/mnt/swx/ThinkVLN/results/watcher_rollout_train_streamvln_rgb_v2

mkdir -p "$DST/images" "$DST/manifest"

cp "$OUT/manifest.jsonl" \
   "$DST/manifest/watcher_rollout_manifest.jsonl"

for f in "$OUT"/shards/*.tar; do
    tar -xf "$f" -C "$DST"
done
