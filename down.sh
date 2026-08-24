# python extract_raw_video_eye_crops.py \
#   --webdav-base-url 'https://webdav.123pan.cn/webdav' \
#   --webdav-username '19957823887' \
#   --webdav-password 'szgswzcg' \
#   --max-edge 256

mkdir -p data/E2E/qinggewang_low_speed_800_all3.compact.parts

curl -L -C - -u '19957823887:j7i9k9a7' \
  'https://webdav.123pan.cn/webdav/qinggewang_low_speed_800_all3.compact.parts/manifest.json' \
  -o 'data/E2E/qinggewang_low_speed_800_all3.compact.parts/manifest.json'

for i in $(seq -f "%04g" 1 13); do
  curl -L -C - -u '19957823887:j7i9k9a7' \
    "https://webdav.123pan.cn/webdav/qinggewang_low_speed_800_all3.compact.parts/part-${i}.jsonl" \
    -o "data/E2E/qinggewang_low_speed_800_all3.compact.parts/part-${i}.jsonl"
done

curl -L -C - -u '19957823887:j7i9k9a7' \
  'https://webdav.123pan.cn/webdav/qinggewang_low_speed_800_all3.selected_rings.jsonl' \
  -o 'data/E2E/qinggewang_low_speed_800_all3.selected_rings.jsonl'

curl -L -C - -u '19957823887:j7i9k9a7' \
  'https://webdav.123pan.cn/webdav/qinggewang_low_speed_800_all3.compact.stats.json' \
  -o 'data/E2E/qinggewang_low_speed_800_all3.compact.stats.json'

curl -L -C - -u '19957823887:j7i9k9a7' \
  'https://webdav.123pan.cn/webdav/README.md' \
  -o 'data/E2E/qinggewang_low_speed_800_all3.README.md'