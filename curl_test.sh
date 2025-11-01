# 1. Encode your image
# IMAGE_B64=$(base64 -w 0 2700x1800.png)
IMAGE_B64=$(base64 -w 0 ocr-s.png)

# 2. Pipe the payload directly to curl using -d @-
#    The shell expands $IMAGE_B64 and the whole block is
#    sent to curl's stdin, not as an argument.
curl http://127.0.0.1:7080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d @- <<EOF
{
  "model": "model",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,$IMAGE_B64"}}
      ]
    }
  ],
  "temperature": 0
}
EOF