import argparse
import time
import numpy as np
import glob
from PIL import Image
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, CalibrationDataReader

# First, prepare the model: python -m onnxruntime.quantization.preprocess --input src/demo2/models/input.onnx --output src/demo2/models/input_preproc.onnx --skip_symbolic_shape True

def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = glob.glob(images_folder + '/*.jpg')
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalizeFactor = 1.0 / 255.0

    for image_name in batch_filenames:
        image_filepath = image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))

        input_data = (np.float32(pillow_img) * normalizeFactor - mean)/std
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


class DataReader(CalibrationDataReader):
    def __init__(self, image_folder: str, model_path: str, size_limit: int):
        self.enum_data = None

        # Use inference session to get input shape
        session = onnxruntime.InferenceSession(model_path, None, providers=['CUDAExecutionProvider'])
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(image_folder, height, width, size_limit=size_limit)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def benchmark(model_path):
    options = onnxruntime.SessionOptions()
    options.profile_enable = True
    # options.log_severity_level = 0
    session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.random.rand(1, 3, 384, 288).astype(np.float32)

    # Warming up
    _ = session.run([], {input_name: input_data})

    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", help="input model", default="src/demo2/models/input_preproc.onnx")
    parser.add_argument("--output_model", help="output model", default="src/demo2/models/optimized/quantized_v1.onnx")
    parser.add_argument("--calibrate_dataset", default="src/demo2/dataset", help="calibration data set")
    parser.add_argument("--quant_format", default=QuantFormat.QDQ, type=QuantFormat.from_string, choices=list(QuantFormat))
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset

    print("Prepare dataset...")
    dr = DataReader(calibration_dataset_path, input_model_path, size_limit=10)

    print("Calibrate and quantize model...")
    # Remark: turn off model optimization during quantization!
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )

    print("Calibrated and quantized model saved.")

    print("Benchmarking fp32 model...")
    benchmark(input_model_path)

    print("Benchmarking int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()
