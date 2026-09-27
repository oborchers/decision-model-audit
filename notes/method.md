# Method notes

## Comparability

The compared systems are not the same kind of thing: 340M to 1B parameter encoders, a closed model of undisclosed size, and decoder LLMs with long context, multimodality and free-text output.

Head-to-head metrics (accuracy, calibration, cost, latency) are computed only on the shared decision contract: identical input, identical label schema, one typed answer per question. Capabilities beyond that contract (context length, modalities, rationale output, local execution, open weights) are reported as a capability matrix, not folded into a score.
