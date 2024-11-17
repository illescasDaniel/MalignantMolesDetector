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
	@State private var malignantProbability: Float?
	@State private var showingImagerPicker: ImagePicker.SourceType?
	@State private var hasError: Bool = false
	@State private var localizedError: LocalizedError?
	private let malignantMolesPredictor = MalignantMolesPredictor()

	var body: some View {
		NavigationStack {
			GeometryReader { geometry in
				VStack {
					Text("Choose an image to detect a malignant mole")
					HStack {
						Button(action: {
							image = nil
							malignantProbability = nil
							localizedError = nil
							hasError = false
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
							malignantProbability = nil
							localizedError = nil
							hasError = false
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

					if let malignantProbability {
						Gauge(value: malignantProbability, in: 0...1) {
							Text("Malignant mole probability")
								.font(.system(.headline))
						} currentValueLabel: {
							Text("\(malignantProbability * 100, specifier: "%.0f")%")
								.font(.largeTitle)
								.foregroundStyle(malignantProbability > 0.8 ? .red : Color(cgColor: UIColor.label.cgColor))
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
		.task {
			do throws(MalignantMolesPredictor.LoadingError) {
				try await malignantMolesPredictor.load()
			} catch {
				localizedError = error
				hasError = true
			}
		}
		.onChange(of: image) { newImage in
			Task {
				guard let newImage else { return }

				do throws(MalignantMolesPredictor.PredictionError) {
					let malignantProbability = try await malignantMolesPredictor.predictMalignantProbability(of: newImage)
					self.malignantProbability = malignantProbability
				} catch {
					localizedError = error
					hasError = true
				}
			}
		}
		.sheet(item: $showingImagerPicker, content: { sourceType in
			ImagePicker(image: $image, sourceType: sourceType)
		})
		.alert("Error", isPresented: $hasError) {
			Button("OK") {}
		} message: {
			Text(localizedError?.errorDescription ?? "Unknown error")
		}

	}
}

#Preview {
	ContentView()
}
