"""ONNX model exporter for deployment."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Exports PyTorch models to ONNX format with verification."""

    def export_model(self, model, dummy_input, output_path: str,
                      model_name: str = "model") -> bool:
        """Export a PyTorch model to ONNX.

        Args:
            model: PyTorch nn.Module
            dummy_input: Example input tensor
            output_path: Output .onnx file path
            model_name: Model name for logging

        Returns:
            True if successful
        """
        try:
            import torch
            import torch.onnx
            import os
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            model.eval()
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            logger.info(f"Exported {model_name} to {output_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    def verify_export(self, original_model, onnx_path: str,
                       dummy_input) -> bool:
        """Verify ONNX model output matches original PyTorch model.

        Args:
            original_model: Original PyTorch model
            onnx_path: Path to exported ONNX file
            dummy_input: Input tensor for comparison

        Returns:
            True if outputs match within tolerance
        """
        try:
            import torch
            import onnxruntime as ort

            original_model.eval()
            with torch.no_grad():
                torch_out = original_model(dummy_input).numpy()

            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            onnx_out = sess.run(None, {input_name: dummy_input.numpy()})[0]

            max_diff = float(np.max(np.abs(torch_out - onnx_out)))
            logger.info(f"ONNX verification max diff: {max_diff:.6f}")
            return max_diff < 1e-4
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False

    def optimize_onnx(self, onnx_path: str, output_path: str) -> bool:
        """Apply basic ONNX graph optimizations.

        Args:
            onnx_path: Input ONNX file
            output_path: Output optimized ONNX file

        Returns:
            True if successful
        """
        try:
            import onnx
            from onnx import optimizer
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            passes = ["eliminate_identity", "eliminate_nop_transpose",
                      "fuse_bn_into_conv", "fuse_consecutive_squeezes"]
            try:
                optimized = optimizer.optimize(model, passes)
            except Exception:
                optimized = model  # Fall back to unoptimized
            onnx.save(optimized, output_path)
            logger.info(f"Saved optimized ONNX to {output_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            return False
