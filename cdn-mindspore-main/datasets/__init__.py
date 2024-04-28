from .hico import build as build_hico
from .hico import preprocess_fn
from .hico_eval import HICOEvaluator

def build_dataset(image_set, args):
    return build_hico(image_set, args)
