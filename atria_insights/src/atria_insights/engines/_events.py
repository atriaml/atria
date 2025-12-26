from ignite.engine import EventEnum


class MetricUpdateEvents(EventEnum):
    X_METRIC_STARTED = "x_metric_started"
    X_METRIC_COMPLETED = "x_metric_completed"
