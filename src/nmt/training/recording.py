from typing import Dict, Optional

class Recorder(object):
    def __init__(self):
        self.metrics = {'lr': []}
        self.batch_metrics = {}

    def record(self, metrics: Dict[str, float], scope: Optional[str] = None):
        for name, value in metrics.items():
            name = f'{scope}/{name}' if scope else name

            if name not in self.batch_metrics:
                self.batch_metrics[name] = []
            self.batch_metrics[name].append(value)

    def stamp(self, step: int = 0, lr: float = None):
        if lr is not None:
            self.metrics['lr'].append((step, lr))
        
        for name, values in self.batch_metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []

            # Add the average of metrics values in the batch.
            self.metrics[name].append((step, sum(values) / len(values)))

        self.batch_metrics.clear()

    def format(self, fstring: str) -> str:
        return fstring.format(**{
            k.replace('/', '_'): v[-1][1] for k, v in self.metrics.items()})
        
    def get_metric(self, metric_name: str) -> float:
      if metric_name in self.metrics:
        return self.metrics[metric_name]
      else:
        raise Exception(f"metric with name f{metric_name} not found in the metrics = f{self.metrics.keys()}")
