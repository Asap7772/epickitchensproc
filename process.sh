gsutil -m rm -rf gs://public-ptr-bucket/toykitchen_numpy_shifted/epic100_bridgeform
python frames_to_bridge_format.py
gsutil -m cp -rn epic100_bridgeform gs://public-ptr-bucket/toykitchen_numpy_shifted/
gsutil -m cp -rn /raid/asap7772/epic100/frames gs://public-ptr-bucket/