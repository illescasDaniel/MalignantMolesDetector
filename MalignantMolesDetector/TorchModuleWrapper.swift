import UIKit
import Accelerate

final class TorchModuleWrapper {
	private var torchModule: TorchModule?

	init?(modelFileName: String) {
		guard let modelFilePath = Bundle.main.path(forResource: modelFileName, ofType: "pt") else {
			print("Model file not found in bundle")
			return nil
		}

		torchModule = TorchModule(fileAtPath: modelFilePath)
		if torchModule == nil {
			print("Failed to initialize TorchModule")
			return nil
		}
	}

	func predict(images: [UIImage]) -> [[NSNumber]]? {
		let targetSize = CGSize(width: 224, height: 224)
		var buffers: [UnsafeMutablePointer<Float>] = []

		for image in images {
			guard let resizedImage = resizeImage(image: image, targetSize: targetSize),
				  let normalizedBuffer = normalizeImage(image: resizedImage) else {
				print("Failed to process image")
				return nil
			}
			buffers.append(normalizedBuffer)
		}

		// Combine all buffers into a single contiguous buffer
		let totalSize = buffers.count * 3 * Int(targetSize.width) * Int(targetSize.height)
		let combinedBuffer = UnsafeMutablePointer<Float>.allocate(capacity: totalSize)

		for (index, buffer) in buffers.enumerated() {
			let offset = index * 3 * Int(targetSize.width) * Int(targetSize.height)
			combinedBuffer.advanced(by: offset).update(from: buffer, count: 3 * Int(targetSize.width) * Int(targetSize.height))
			buffer.deallocate() // Deallocate individual buffers after copying
		}

		// Call the batch prediction method
		guard let results = torchModule?.predictImages(combinedBuffer, numberOfImages: Int32(images.count)) else {
			print("Failed to predict images")
			combinedBuffer.deallocate() // Ensure deallocation on failure
			return nil
		}

		combinedBuffer.deallocate() // Deallocate after use

		// Convert results to probabilities using softmax
		var probabilities: [[NSNumber]] = []
		for result in results {
			if let logits = result as? [Float] {
				let softmaxed = softmax(logits)
				probabilities.append(softmaxed.map { NSNumber(value: $0) })
			}
		}

		return probabilities
	}

	// MARK: Private

	private func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage? {
		UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
		image.draw(in: CGRect(origin: .zero, size: targetSize))
		let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
		UIGraphicsEndImageContext()
		return resizedImage
	}

	private func normalizeImage(image: UIImage) -> UnsafeMutablePointer<Float>? {
		guard let cgImage = image.cgImage else { return nil }

		let width = cgImage.width
		let height = cgImage.height
		let bytesPerPixel = 4
		let bytesPerRow = bytesPerPixel * width
		let bitsPerComponent = 8

		guard let colorSpace = cgImage.colorSpace else { return nil }
		let rawData = UnsafeMutablePointer<UInt8>.allocate(capacity: height * bytesPerRow)
		defer { rawData.deallocate() }

		guard let context = CGContext(data: rawData,
									  width: width,
									  height: height,
									  bitsPerComponent: bitsPerComponent,
									  bytesPerRow: bytesPerRow,
									  space: colorSpace,
									  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
			return nil
		}

		context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

		let floatBuffer = UnsafeMutablePointer<Float>.allocate(capacity: 3 * width * height)

		for y in 0..<height {
			for x in 0..<width {
				let pixelIndex = (y * width + x) * bytesPerPixel
				let r = Float(rawData[pixelIndex]) / 255.0
				let g = Float(rawData[pixelIndex + 1]) / 255.0
				let b = Float(rawData[pixelIndex + 2]) / 255.0

				// Normalize the pixel values to the range expected by the model
				floatBuffer[y * width + x] = (r - 0.485) / 0.229
				floatBuffer[width * height + y * width + x] = (g - 0.456) / 0.224
				floatBuffer[2 * width * height + y * width + x] = (b - 0.406) / 0.225
			}
		}

		return floatBuffer
	}

	private func softmax(_ logits: [Float]) -> [Float] {
		let maxLogit = logits.max() ?? 0
		let exps = logits.map { exp($0 - maxLogit) }
		let sumExps = exps.reduce(0, +)
		return exps.map { $0 / sumExps }
	}
}
