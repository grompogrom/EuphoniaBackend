import glob


def find_file(name):
    files = glob.glob(f'**/{name}', recursive=True)
    for file in files:
        return file


if __name__ == '__main__':
    f1 = find_file("model_base_touhou.onnx")
    f2 = find_file("model_token_touhou.onnx")
    print(f1)
    print(f2)
