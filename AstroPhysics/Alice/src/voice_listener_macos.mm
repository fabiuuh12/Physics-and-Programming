#import <AVFoundation/AVFoundation.h>
#import <Speech/Speech.h>

#include "alice/voice_listener.hpp"
#include "alice/string_utils.hpp"

#include <chrono>
#include <thread>

namespace alice {

struct VoiceListener::Impl {
    bool is_available = false;
    std::string backend = "none";
    std::string error;
    SFSpeechRecognizer* recognizer = nil;
};

static bool wait_for_bool_async(const std::function<void(void (^)(bool))>& trigger, bool& out_value,
                                double timeout_seconds = 8.0) {
    __block bool done = false;
    __block bool value = false;

    trigger(^(bool result) {
      value = result;
      done = true;
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int>(timeout_seconds * 1000.0));
    while (!done && std::chrono::steady_clock::now() < deadline) {
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.02]];
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!done) {
        return false;
    }
    out_value = value;
    return true;
}

VoiceListener::VoiceListener() : impl_(new Impl()) {
    impl_->backend = "speech";

    bool speech_ok = false;
    bool mic_ok = false;

    bool speech_waited = wait_for_bool_async(
        [](void (^cb)(bool)) {
            [SFSpeechRecognizer requestAuthorization:^(SFSpeechRecognizerAuthorizationStatus status) {
              cb(status == SFSpeechRecognizerAuthorizationStatusAuthorized);
            }];
        },
        speech_ok);

    bool mic_waited = wait_for_bool_async(
        [](void (^cb)(bool)) {
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
              cb(granted == YES);
            }];
        },
        mic_ok);

    if (!speech_waited) {
        impl_->error = "Speech authorization timed out.";
        return;
    }
    if (!speech_ok) {
        impl_->error = "Speech recognition permission denied.";
        return;
    }
    if (!mic_waited) {
        impl_->error = "Microphone permission request timed out.";
        return;
    }
    if (!mic_ok) {
        impl_->error = "Microphone permission denied.";
        return;
    }

    NSLocale* locale = [NSLocale localeWithLocaleIdentifier:@"en-US"];
    impl_->recognizer = [[SFSpeechRecognizer alloc] initWithLocale:locale];
    if (impl_->recognizer == nil) {
        impl_->error = "Unable to initialize speech recognizer.";
        return;
    }

    impl_->is_available = true;
}

VoiceListener::~VoiceListener() {
    delete impl_;
}

bool VoiceListener::available() const {
    return impl_->is_available;
}

std::string VoiceListener::backend_name() const {
    return impl_->backend;
}

std::string VoiceListener::last_error() const {
    return impl_->error;
}

std::optional<std::string> VoiceListener::listen(double timeout_seconds, double phrase_time_limit_seconds,
                                                 const std::function<void()>& tick) {
    if (!impl_->is_available || impl_->recognizer == nil) {
        return std::nullopt;
    }

    AVAudioEngine* engine = [[AVAudioEngine alloc] init];
    SFSpeechAudioBufferRecognitionRequest* request = [[SFSpeechAudioBufferRecognitionRequest alloc] init];
    request.shouldReportPartialResults = YES;

    __block NSString* best_text = nil;
    __block bool done = false;
    __block NSError* task_error = nil;
    __block bool has_speech = false;
    __block auto speech_started_at = std::chrono::steady_clock::time_point{};

    SFSpeechRecognitionTask* task =
        [impl_->recognizer recognitionTaskWithRequest:request
                                        resultHandler:^(SFSpeechRecognitionResult* result, NSError* error) {
                                          if (result != nil) {
                                              NSString* text = result.bestTranscription.formattedString;
                                              if (text != nil && text.length > 0) {
                                                  best_text = text;
                                                  if (!has_speech) {
                                                      has_speech = true;
                                                      speech_started_at = std::chrono::steady_clock::now();
                                                  }
                                              }
                                              if (result.isFinal) {
                                                  done = true;
                                              }
                                          }
                                          if (error != nil) {
                                              task_error = error;
                                              done = true;
                                          }
                                        }];

    AVAudioInputNode* input_node = engine.inputNode;
    if (input_node == nil) {
        impl_->error = "No microphone input node available.";
        [task cancel];
        return std::nullopt;
    }

    AVAudioFormat* format = [input_node outputFormatForBus:0];
    [input_node installTapOnBus:0
                     bufferSize:1024
                         format:format
                          block:^(AVAudioPCMBuffer* buffer, AVAudioTime* when) {
                            (void)when;
                            [request appendAudioPCMBuffer:buffer];
                          }];

    NSError* start_error = nil;
    if (![engine startAndReturnError:&start_error]) {
        [input_node removeTapOnBus:0];
        [task cancel];
        impl_->error = start_error != nil ? std::string([[start_error localizedDescription] UTF8String])
                                          : "Failed to start audio engine.";
        return std::nullopt;
    }

    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + std::chrono::milliseconds(static_cast<int>(timeout_seconds * 1000.0));

    while (!done) {
        if (tick) {
            tick();
        }

        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.02]];

        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            break;
        }
        if (has_speech && phrase_time_limit_seconds > 0.0) {
            const auto phrase_deadline = speech_started_at + std::chrono::milliseconds(static_cast<int>(phrase_time_limit_seconds * 1000.0));
            if (now >= phrase_deadline) {
                break;
            }
        }
    }

    [engine stop];
    [input_node removeTapOnBus:0];
    [request endAudio];
    [task cancel];

    if (task_error != nil && (best_text == nil || best_text.length == 0)) {
        impl_->error = std::string([[task_error localizedDescription] UTF8String]);
        return std::nullopt;
    }

    if (best_text == nil || best_text.length == 0) {
        return std::nullopt;
    }

    std::string text([best_text UTF8String]);
    text = trim(text);
    if (text.empty()) {
        return std::nullopt;
    }
    return text;
}

}  // namespace alice
