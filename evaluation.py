from dataclasses import dataclass

@dataclass(frozen=True)
class EvaluationResult:
	accuracy: float

def evaluate_model(model, data) -> EvaluationResult:
	total_count = 0
	correct_count = 0
	for instance in data:
		total_count += 1
		prediction = model.predict(instance)
		if prediction == instance.label:
			correct_count += 1

	accuracy = correct_count / total_count
	return EvaluationResult(accuracy=accuracy)
