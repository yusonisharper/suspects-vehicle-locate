# locate suspects vehicle via edge devices

Usage: 
Download the entire file, and use the following command to start detection. \
\
`python3 detectnet.py --model=./networks/az_ocr/az_ocr_ssdmobilenetv1_2.onnx --class_labels=./networks/az_ocr/labels.txt --input_blob=input_0 --output_cvg=scores --output_bbox=boxes --camera=csi://0 --width=740  --height=480` \
Where `--camera=csi://0` is your camera. If you are using other cameras like USB cam, replace `csi://0` with the appropriate address.
