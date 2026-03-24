"""TFLite converter for edge deployment."""
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class TFLiteConverter:
    """Converts models to TensorFlow Lite format."""

    def convert_from_onnx(self, onnx_path: str, output_path: str) -> bool:
        """Convert ONNX model to TFLite via onnx-tf.

        Args:
            onnx_path: Path to ONNX model
            output_path: Output .tflite path

        Returns:
            True if successful
        """
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
            import os
            import tempfile

            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tmp_dir = tempfile.mkdtemp()
            tf_rep.export_graph(tmp_dir)

            converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
            tflite_model = converter.convert()

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(tflite_model)
            logger.info(f"TFLite model saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"TFLite conversion from ONNX failed: {e}")
            return False

    def convert_from_pytorch(self, model, dummy_input,
                               output_path: str) -> bool:
        """Convert PyTorch model → ONNX → TFLite.

        Args:
            model: PyTorch nn.Module
            dummy_input: Example input tensor
            output_path: Output .tflite path

        Returns:
            True if successful
        """
        import tempfile, os
        from deployment.onnx_exporter import ONNXExporter

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_tmp = f.name

        exporter = ONNXExporter()
        if not exporter.export_model(model, dummy_input, onnx_tmp):
            return False
        result = self.convert_from_onnx(onnx_tmp, output_path)
        try:
            os.unlink(onnx_tmp)
        except OSError:
            pass
        return result

    def quantize(self, tflite_path: str, output_path: str,
                  quantization: str = "int8") -> bool:
        """Apply post-training quantization to a TFLite model.

        Args:
            tflite_path: Input .tflite path
            output_path: Output quantized .tflite path
            quantization: 'int8' or 'fp16'

        Returns:
            True if successful
        """
        try:
            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_saved_model(tflite_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if quantization == "fp16":
                converter.target_spec.supported_types = [tf.float16]
            tflite_quant = converter.convert()
            with open(output_path, "wb") as f:
                f.write(tflite_quant)
            logger.info(f"Quantized TFLite saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"TFLite quantization failed: {e}")
            return False

    def benchmark(self, tflite_path: str, input_shape: tuple = (1, 50, 3),
                   n_runs: int = 100) -> dict:
        """Benchmark TFLite model inference latency.

        Args:
            tflite_path: Path to .tflite model
            input_shape: Input tensor shape
            n_runs: Number of inference runs

        Returns:
            Dict with mean, min, max latency_ms
        """
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            dummy = np.random.randn(*input_shape).astype(np.float32)
            latencies = []
            for _ in range(n_runs):
                interpreter.set_tensor(input_details[0]["index"], dummy)
                t0 = time.perf_counter()
                interpreter.invoke()
                latencies.append((time.perf_counter() - t0) * 1000.0)

            return {
                "mean_ms": float(np.mean(latencies)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "p99_ms": float(np.percentile(latencies, 99)),
            }
        except Exception as e:
            logger.error(f"TFLite benchmark failed: {e}")
            return {}
