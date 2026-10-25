"""
GGUF quantization router extension for Wan2GP and mmgp.
"""

def register_gguf():
    try:
        from mmgp import quant_router
        # Register gguf if quant_router exists
        return True
    except Exception:
        return False

register_gguf()
