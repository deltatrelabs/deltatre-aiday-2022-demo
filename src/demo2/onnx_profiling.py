import argparse
import time
import numpy as np
import onnxruntime

def profile(model_path):
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    # options.log_severity_level = 0

    session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name

    input_data = np.random.rand(1, 3, 384, 288).astype(np.float32)

    # Warm-up
    _ = session.run([], {input_name: input_data})

    start = time.perf_counter()
    _ = session.run([], {input_name: input_data})
    end = (time.perf_counter() - start) * 1000

    prof_file = session.end_profiling()

    print(f"Inference time: {end:.2f}ms")
    print(prof_file)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", help="input model", default="src/demo2/models/optimized/quantized_v1.onnx")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model

    print("Profiling int8 model...")
    profile(input_model_path)


if __name__ == "__main__":
    main()
