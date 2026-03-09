#import <AVFoundation/AVFoundation.h>
#import <Vision/Vision.h>

#include "alice/face_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>

struct AliceFaceTrackerState {
    std::mutex* mutex = nullptr;
    alice::FaceObservation* observation = nullptr;
    std::string* error = nullptr;
};

static bool wait_for_bool_async(const std::function<void(void (^)(bool))>& trigger, bool& out_value,
                                double timeout_seconds = 8.0) {
    __block bool done = false;
    __block bool value = false;

    trigger(^(bool result) {
      value = result;
      done = true;
    });

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(static_cast<int>(timeout_seconds * 1000.0));
    while (!done && std::chrono::steady_clock::now() < deadline) {
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode
                                 beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.02]];
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!done) {
        return false;
    }
    out_value = value;
    return true;
}

static VNFaceObservation* select_best_face(NSArray<VNFaceObservation*>* faces) {
    if (faces == nil || faces.count == 0) {
        return nil;
    }
    VNFaceObservation* best = faces[0];
    CGFloat best_area = best.boundingBox.size.width * best.boundingBox.size.height;
    for (VNFaceObservation* face in faces) {
        const CGRect box = face.boundingBox;
        const CGFloat area = box.size.width * box.size.height;
        if (area > best_area) {
            best = face;
            best_area = area;
        }
    }
    return best;
}

@interface AliceFaceCaptureDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(nonatomic, assign) void* owner;
@end

@implementation AliceFaceCaptureDelegate

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
    (void)output;
    (void)connection;

    AliceFaceTrackerState* state = static_cast<AliceFaceTrackerState*>(self.owner);
    if (state == nullptr || state->mutex == nullptr || state->observation == nullptr) {
        return;
    }

    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (imageBuffer == nullptr) {
        return;
    }

    @autoreleasepool {
        VNDetectFaceRectanglesRequest* request = [[VNDetectFaceRectanglesRequest alloc] init];
        VNImageRequestHandler* handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:imageBuffer
                                                                                   orientation:kCGImagePropertyOrientationUp
                                                                                       options:@{}];

        NSError* error = nil;
        const BOOL ok = [handler performRequests:@[ request ] error:&error];

        alice::FaceObservation obs;
        if (ok && request.results != nil && request.results.count > 0) {
            NSArray<VNFaceObservation*>* faces = (NSArray<VNFaceObservation*>*)request.results;
            VNFaceObservation* best = select_best_face(faces);
            if (best != nil) {
                const CGRect box = best.boundingBox;
                const float cx = static_cast<float>(box.origin.x + box.size.width * 0.5);
                const float cy = static_cast<float>(box.origin.y + box.size.height * 0.5);

                obs.found = true;
                obs.face_count = static_cast<int>(faces.count);
                obs.x = std::clamp((cx - 0.5f) * 2.0f, -1.0f, 1.0f);
                obs.y = std::clamp((0.5f - cy) * 2.0f, -1.0f, 1.0f);
            }
        }

        {
            std::scoped_lock lock(*state->mutex);
            *state->observation = obs;
            if (!ok && error != nil && state->error != nullptr) {
                *state->error = std::string([[error localizedDescription] UTF8String]);
            }
        }
    }
}

@end

namespace alice {

struct FaceTracker::Impl {
    mutable std::mutex mutex;
    bool is_running = false;
    std::string error;
    FaceObservation observation;

    AVCaptureSession* session = nil;
    dispatch_queue_t capture_queue = nil;
    id delegate = nil;
    AliceFaceTrackerState delegate_state;
};

FaceTracker::FaceTracker() : impl_(new Impl()) {}

FaceTracker::~FaceTracker() {
    stop();
    delete impl_;
}

bool FaceTracker::start(int camera_index) {
    if (impl_->is_running) {
        return true;
    }

    @try {
        bool camera_ok = false;
        const bool waited = wait_for_bool_async(
            [](void (^cb)(bool)) {
                [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                                         completionHandler:^(BOOL granted) {
                                           cb(granted == YES);
                                         }];
            },
            camera_ok);

        if (!waited) {
            impl_->error = "Camera permission request timed out.";
            return false;
        }
        if (!camera_ok) {
            impl_->error = "Camera permission denied.";
            return false;
        }

        AVCaptureDeviceDiscoverySession* discovery =
            [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[
                AVCaptureDeviceTypeBuiltInWideAngleCamera,
                AVCaptureDeviceTypeExternal
            ]
                                                                  mediaType:AVMediaTypeVideo
                                                                   position:AVCaptureDevicePositionUnspecified];

        NSArray<AVCaptureDevice*>* devices = discovery.devices;
        if (devices == nil || devices.count == 0) {
            impl_->error = "No camera device found.";
            return false;
        }

        NSInteger index = camera_index;
        if (index < 0 || index >= devices.count) {
            index = 0;
        }

        AVCaptureDevice* device = devices[index];
        NSError* input_error = nil;
        AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&input_error];
        if (input == nil) {
            impl_->error = input_error != nil ? std::string([[input_error localizedDescription] UTF8String])
                                              : "Unable to create camera input.";
            return false;
        }

        impl_->session = [[AVCaptureSession alloc] init];
        [impl_->session beginConfiguration];
        if ([impl_->session canAddInput:input]) {
            [impl_->session addInput:input];
        } else {
            [impl_->session commitConfiguration];
            impl_->error = "Cannot add selected camera input to capture session.";
            return false;
        }

        AVCaptureVideoDataOutput* output = [[AVCaptureVideoDataOutput alloc] init];
        output.alwaysDiscardsLateVideoFrames = YES;
        output.videoSettings = @{ (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) };

        impl_->capture_queue = dispatch_queue_create("com.fabiofacin.alice.facecapture", DISPATCH_QUEUE_SERIAL);
        AliceFaceCaptureDelegate* delegate = [[AliceFaceCaptureDelegate alloc] init];

        impl_->delegate_state.mutex = &impl_->mutex;
        impl_->delegate_state.observation = &impl_->observation;
        impl_->delegate_state.error = &impl_->error;

        delegate.owner = &impl_->delegate_state;
        impl_->delegate = delegate;
        [output setSampleBufferDelegate:delegate queue:impl_->capture_queue];

        if ([impl_->session canAddOutput:output]) {
            [impl_->session addOutput:output];
        }

        [impl_->session commitConfiguration];
        [impl_->session startRunning];

        impl_->is_running = true;
        return true;
    } @catch (NSException* ex) {
        impl_->error = std::string("Face tracker exception: ") + [[ex reason] UTF8String];
        return false;
    }
}

void FaceTracker::stop() {
    if (!impl_->is_running) {
        return;
    }

    @try {
        if (impl_->session != nil) {
            [impl_->session stopRunning];
        }
    } @catch (...) {
    }

    impl_->session = nil;
    impl_->delegate = nil;
    impl_->capture_queue = nil;

    {
        std::scoped_lock lock(impl_->mutex);
        impl_->observation = FaceObservation{};
    }
    impl_->is_running = false;
}

bool FaceTracker::running() const { return impl_->is_running; }

std::string FaceTracker::last_error() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->error;
}

FaceObservation FaceTracker::latest() const {
    std::scoped_lock lock(impl_->mutex);
    return impl_->observation;
}

}  // namespace alice
