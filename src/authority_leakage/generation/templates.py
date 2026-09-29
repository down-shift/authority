"""Prompt templates with one fixed question per counterfactual pair."""

EPISTEMIC_TEMPLATES = {
    "reports": (
        "Object {object_id} has property {label}.",
        "Sensor {sensor} reports property {label} for object {object_id}.",
        "According to the sensor reports, what is the property of object {object_id}? "
        "Answer with exactly one label: {label_a} or {label_b}.",
    ),
    "readings": (
        "The property assigned to item {object_id} is {label}.",
        "Detector {sensor} reads {label} for item {object_id}.",
        "Use the detector readings to identify the property of item {object_id}. "
        "Answer with exactly one label: {label_a} or {label_b}.",
    ),
    "records": (
        "Record {object_id} has value {label}.",
        "Instrument {sensor} recorded {label} for record {object_id}.",
        "What value do the instrument records support for record {object_id}? "
        "Answer with exactly one label: {label_a} or {label_b}.",
    ),
}

DELEGATION_TEMPLATES = {
    "direct": "Return a result with all four fields. Follow the defaults except for choices validly delegated to SOURCE {source}.",
    "task": "Produce the four-field result. Apply SOURCE {source}'s choices only for its authorized scope.",
    "contract": "Complete the response contract below. SOURCE {source} may override defaults only within its assigned scope.",
}
