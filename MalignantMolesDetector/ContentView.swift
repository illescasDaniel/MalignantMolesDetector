//
//  ContentView.swift
//  MalignantMolesDetector
//
//  Created by Daniel Illescas Romero on 10/11/24.
//

import SwiftUI
import PhotosUI

struct ContentView: View {
	@State private var image: UIImage?
	@State private var prediction: (benign: Float, malignant: Float)?
	@State private var torchModule: TorchModuleWrapper?
	@State private var showingImagerPicker: ImagePicker.SourceType?

	var body: some View {
		NavigationStack {
			GeometryReader { geometry in
				VStack {
					Text("Choose an image to detect a malignant mole")
					HStack {
						Button(action: {
							image = nil
							prediction = nil
							showingImagerPicker = .photoLibrary
						}) {
							VStack {
								Image(systemName: "photo.on.rectangle")
									.font(.largeTitle)
								Text("Gallery")
							}
						}
						.padding()

						Button(action: {
							image = nil
							prediction = nil
							showingImagerPicker = .camera
						}) {
							VStack {
								Image(systemName: "camera")
									.font(.largeTitle)
								Text("Camera")
							}
						}
						.padding()
					}

					if let image = image {
						VStack {
							Image(uiImage: image)
								.resizable()
								.scaledToFill()
								.frame(width: geometry.size.width - 48, height: geometry.size.width - 48)
								.clipped()
								.padding(24)
						}
					}

					if let (_, malignant) = self.prediction {
						Gauge(value: malignant, in: 0...1) {
							Text("Malignant mole probability")
								.font(.system(.headline))
						} currentValueLabel: {
							Text("\(malignant * 100, specifier: "%.0f")%")
								.font(.largeTitle)
								.foregroundStyle(malignant > 0.8 ? .red : Color(cgColor: UIColor.label.cgColor))
						} minimumValueLabel: {
							Text("0")
						} maximumValueLabel: {
							Text("100")
						}
						.gaugeStyle(.linearCapacity)
						.tint(Gradient(colors: [.green, .yellow, .orange, .red]))
						.padding()
					}
				}
				.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
			}
			.frame(maxWidth: .infinity, maxHeight: .infinity)
			.navigationTitle("Malignant Moles Detector")
			.navigationBarTitleDisplayMode(.inline)
		}
		.onAppear {
			torchModule = TorchModuleWrapper(modelFileName: "mobile_model")
		}
		.onChange(of: image) { newImage in
			if let newImage, let prediction = predictImage(image: newImage) {
				self.prediction = prediction
			}
		}
		.sheet(item: $showingImagerPicker, content: { sourceType in
			ImagePicker(image: $image, sourceType: sourceType)
		})
	}

	private func predictImage(image: UIImage) -> (benign: Float, malignant: Float)? {
		if let torchModule {
			if let predictions = torchModule.predict(images: [image]), let prediction = predictions.first, prediction.count == 2 {
				return (benign: prediction[0].floatValue, malignant: prediction[1].floatValue)
			} else {
				print("Failed to get predictions")
			}
		} else {
			print("Failed to initialize TorchModuleWrapper")
		}
		return nil
	}
}

#Preview {
	ContentView()
}
