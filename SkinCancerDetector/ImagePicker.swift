import SwiftUI

struct ImagePicker: UIViewControllerRepresentable {

	enum SourceType: Int, Identifiable {
		var id: Int {
			rawValue
		}
		case camera
		case photoLibrary
	}

	@Binding var image: UIImage?
	let sourceType: SourceType
	
	@Environment(\.presentationMode) var presentationMode

	final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
		var parent: ImagePicker

		init(parent: ImagePicker) {
			self.parent = parent
		}

		func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
			if let uiImage = info[.editedImage] as? UIImage {
				parent.image = uiImage
			} else if let uiImage = info[.originalImage] as? UIImage {
				parent.image = uiImage
			}
			parent.presentationMode.wrappedValue.dismiss()
		}

		func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
			parent.presentationMode.wrappedValue.dismiss()
		}
	}

	func makeCoordinator() -> Coordinator {
		Coordinator(parent: self)
	}

	func makeUIViewController(context: Context) -> UIImagePickerController {
		let picker = UIImagePickerController()
		picker.delegate = context.coordinator
		picker.allowsEditing = true
		switch sourceType {
		case .camera:
			picker.sourceType = .camera
		case .photoLibrary:
			picker.sourceType = .photoLibrary
		}
		return picker
	}

	func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}
