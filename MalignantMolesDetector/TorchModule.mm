#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation TorchModule {
@protected
	torch::jit::mobile::Module _impl;
}

- (instancetype)init {
	self = [super init];
	return self;
}

- (BOOL)loadFileAtPath:(NSString*)filePath error:(NSError**)error NS_REFINED_FOR_SWIFT {
	try {
		_impl = torch::jit::_load_for_mobile(filePath.UTF8String);
		return YES;
	} catch (const std::exception& exception) {
		if (error) {
			NSString *errorMessage = [NSString stringWithUTF8String:exception.what()];
			*error = [NSError errorWithDomain:@"TorchLoadErrorDomain"
										 code:-1
									 userInfo:@{NSLocalizedDescriptionKey: errorMessage}];
		}
		return NO;
	}
}

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
		int64_t numClasses = outputTensor.size(1);

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
