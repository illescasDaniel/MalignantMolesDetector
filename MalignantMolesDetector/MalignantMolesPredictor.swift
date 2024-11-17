import UIKit

actor MalignantMolesPredictor {

	typealias LoadingError = TorchModuleWrapper.LoadingError

	enum PredictionError: LocalizedError {
		case emptyPrediction
		case incorrectPredictionsCount(Int)
		case incorrectValidPredictionsCount(Int)
		case incorrectTargetClassesCount(Int)

		var errorDescription: String? {
			switch self {
			case .emptyPrediction: return "Empty prediction"
			case .incorrectPredictionsCount(let count): return "Incorrect predictions count: \(count)"
			case .incorrectValidPredictionsCount(let count): return "Incorrect valid predictions count: \(count)"
			case .incorrectTargetClassesCount(let count): return "Incorrect target classes count: \(count)"
			}
		}
	}

	private let torchModuleWrapper = TorchModuleWrapper()

	init() {}

	func load() async throws(LoadingError) {
		try await torchModuleWrapper.load(modelFileName: "mobile_model")
	}

	/// Returns malignant mole probability percent normalized between 0 and 1
	func predictMalignantProbability(of image: UIImage) async throws(PredictionError) -> Float {
		let predictions: [(benign: Float, malignant: Float)] = try await predict(images: [image])
		guard predictions.count == 1 else {
			throw .incorrectPredictionsCount(predictions.count)
		}
		return predictions[0].malignant
	}

	func predict(images: [UIImage]) async throws(PredictionError) -> [(benign: Float, malignant: Float)] {
		guard let predictions: [[NSNumber]] = await torchModuleWrapper.predict(images: images, targetSize: CGSize(width: 224, height: 224)) else {
			throw .emptyPrediction
		}
		guard images.count == predictions.count else {
			throw .incorrectPredictionsCount(predictions.count)
		}

		var valid_predictions: [(benign: Float, malignant: Float)] = []
		for prediction in predictions {
			if prediction.count != 2 {
				throw .incorrectTargetClassesCount(prediction.count)
			}
			valid_predictions.append((prediction[0].floatValue, prediction[1].floatValue))
		}

		guard images.count == valid_predictions.count else {
			throw .incorrectValidPredictionsCount(valid_predictions.count)
		}

		return valid_predictions
	}
}
