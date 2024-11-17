#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

/// Initializes the object without loading a file.
/// Use `loadFileAtPath:error:` to load the file after initialization.
- (instancetype)init NS_DESIGNATED_INITIALIZER;

/// Loads a Torch model file from the given file path.
/// Returns `YES` if successful or `NO` and sets an `NSError` if an error occurs.
/// Refined to a throwing function in Swift.
- (BOOL)loadFileAtPath:(NSString *)filePath error:(NSError * _Nullable * _Nullable)error;

/// Makes predictions on a buffer of images.
/// @param imageBuffer A pointer to the image data.
/// @param N The number of images.
/// @return An array of predictions, where each prediction is an array of numbers.
- (NSArray<NSArray<NSNumber *> *> *)predictImages:(void *)imageBuffer numberOfImages:(int)N;

/// Unavailable initializers.
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
