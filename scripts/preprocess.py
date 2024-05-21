import tempfile

import apache_beam as beam
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam


def preprocessing_fn(inputs):
    x = inputs['x']
    x = tft.scale_to_0_1(x)
    return {'x': x}


def run_pipeline(raw_data):
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            raw_data = pipeline | beam.Create(raw_data)
            transformed_dataset, transform_fn = (
                (raw_data, None)
                | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
            )
            transformed_data = transformed_dataset | beam.Map(print)
    return transformed_data


if __name__ == '__main__':
    raw_data = [{'x': 1}, {'x': 2}, {'x': 3}]
    run_pipeline(raw_data)
