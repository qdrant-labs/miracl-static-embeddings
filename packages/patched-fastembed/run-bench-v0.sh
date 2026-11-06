SOURCE_FOLDER=$1

cd /tmp && PYTHONPATH=${SOURCE_FOLDER}/packages/patched-fastembed/src uv run --no-config --isolated \
    --no-project --with 'tokenizers<1' --with numpy --with pytest --with onnxruntime \
    --with tqdm --with requests --with huggingface-hub --with loguru --with pillow \
    --with mmh3 --with py-rust-stemmers \
    pytest ${SOURCE_FOLDER}/packages/patched-fastembed/tests -s -p no:cacheprovider
