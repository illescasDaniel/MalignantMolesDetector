#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation TorchModule {
@protected
	torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
	self = [super init];
	if (self) {
		try {
			_impl = torch::jit::_load_for_mobile(filePath.UTF8String);
		} catch (const std::exception& exception) {
			NSLog(@"%s", exception.what());
			return nil;
		}
	}
	return self;
}

//- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
//	try {
//		at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 224, 224}, at::kFloat);
//		c10::InferenceMode guard;
//		auto outputTensor = _impl.forward({tensor}).toTensor();
//		float* floatBuffer = outputTensor.data_ptr<float>();
//		if (!floatBuffer) {
//			return nil;
//		}
//		NSMutableArray* results = [[NSMutableArray alloc] init];
//		for (int i = 0; i < 1000; i++) {
//			[results addObject:@(floatBuffer[i])];
//		}
//		return [results copy];
//	} catch (const std::exception& exception) {
//		NSLog(@"%s", exception.what());
//	}
//	return nil;
//}

- (NSArray<NSArray<NSNumber*>*>*)predictImages:(void*)imageBuffer numberOfImages:(int)N {
	try {
		// Create a tensor from the image buffer with batch size N
		at::Tensor tensor = torch::from_blob(imageBuffer, {N, 3, 224, 224}, at::kFloat);

		// Ensure inference mode is enabled
		c10::InferenceMode guard;

		// Forward pass through the model
		auto outputTensor = _impl.forward({tensor}).toTensor();

		// Check if the output tensor is valid
		if (!outputTensor.defined()) {
			return nil;
		}

		// Assuming the output is a 2D tensor with shape [N, num_classes]
		int numClasses = outputTensor.size(1);

		// Prepare to extract results
		NSMutableArray* batchResults = [[NSMutableArray alloc] init];

		// Iterate over each image in the batch
		for (int i = 0; i < N; i++) {
			// Get the result for the i-th image
			auto imageOutput = outputTensor[i];
			float* floatBuffer = imageOutput.data_ptr<float>();

			// Collect results for this image
			NSMutableArray* imageResults = [[NSMutableArray alloc] init];
			for (int j = 0; j < numClasses; j++) {
				[imageResults addObject:@(floatBuffer[j])];
			}

			// Add the results for this image to the batch results
			[batchResults addObject:[imageResults copy]];
		}

		return [batchResults copy];
	} catch (const std::exception& exception) {
		NSLog(@"%s", exception.what());
	}
	return nil;
}


@end
