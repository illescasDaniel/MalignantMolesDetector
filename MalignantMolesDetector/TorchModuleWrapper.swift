//
//  TorchModuleWrapper.swift
//  MalignantMolesDetector
//
//  Created by Daniel Illescas Romero on 17/11/24.
//

import UIKit

actor TorchModuleWrapper {

	enum LoadingError: LocalizedError {
		case modelNotFound
		case torchModuleLoadingError(Error)

		var errorDescription: String? {
			switch self {
			case .modelNotFound: return "Model not found"
			case .torchModuleLoadingError(let error): return error.localizedDescription
			}
		}
	}

	private let torchModule = TorchModule()

	init() {}

	func load(modelFileName: String) async throws(LoadingError) {
		guard let modelFilePath = Bundle.main.path(forResource: modelFileName, ofType: "pt") else {
			throw .modelNotFound
		}
		do {
			try await Task { try torchModule.loadFile(atPath: modelFilePath) }.value
		} catch {
			throw .torchModuleLoadingError(error)
		}
	}

	func predict(images: [UIImage], targetSize: CGSize) async -> [[NSNumber]]? {
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
		let results = await Task { torchModule.predictImages(combinedBuffer, numberOfImages: Int32(images.count)) }.value

		combinedBuffer.deallocate() // Deallocate after use

		return results // softmaxed probabilities
	}

	// MARK: - Private Methods

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
}
