#import <AVFoundation/AVFoundation.h>
#import <Speech/Speech.h>

#include "alice/voice_listener.hpp"
#include "alice/string_utils.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
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
    @try {
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

        NSString* localeId = @"en-US";
        if (const char* envLocale = std::getenv("ALICE_STT_LOCALE"); envLocale != nullptr && envLocale[0] != '\0') {
            localeId = [NSString stringWithUTF8String:envLocale];
        }
        NSLocale* locale = [NSLocale localeWithLocaleIdentifier:localeId];
        impl_->recognizer = [[SFSpeechRecognizer alloc] initWithLocale:locale];
        if (impl_->recognizer == nil) {
            impl_->error = "Unable to initialize speech recognizer.";
            return;
        }

        impl_->is_available = true;
    } @catch (NSException* ex) {
        impl_->is_available = false;
        impl_->backend = "none";
        impl_->error = std::string("Speech init exception: ") + [[ex reason] UTF8String];
    }
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
                                                 const std::function<void()>& tick,
                                                 const std::function<void(const std::string&)>& on_partial_text) {
    if (!impl_->is_available || impl_->recognizer == nil) {
        return std::nullopt;
    }
    impl_->error.clear();

    @try {
        if ([impl_->recognizer isAvailable] == NO) {
            impl_->error = "Speech recognizer is currently unavailable.";
            return std::nullopt;
        }
        AVAudioEngine* engine = [[AVAudioEngine alloc] init];
        SFSpeechAudioBufferRecognitionRequest* request = [[SFSpeechAudioBufferRecognitionRequest alloc] init];
        // Keep partial hypotheses so we can still return recognized speech on timeout.
        request.shouldReportPartialResults = YES;
        request.taskHint = SFSpeechRecognitionTaskHintDictation;
        if ([request respondsToSelector:@selector(setRequiresOnDeviceRecognition:)]) {
            if (const char* raw = std::getenv("ALICE_STT_ON_DEVICE"); raw != nullptr && raw[0] != '\0') {
                const std::string value = to_lower(trim(raw));
                if (value == "1" || value == "true" || value == "yes" || value == "on") {
                    request.requiresOnDeviceRecognition = YES;
                }
            }
        }

        struct RecognitionState {
            std::mutex mutex;
            std::string best_text;
            std::string task_error;
            bool done = false;
            bool has_speech = false;
            std::chrono::steady_clock::time_point speech_started_at{};
        };
        const auto state = std::make_shared<RecognitionState>();
        const auto partial_callback = std::make_shared<std::function<void(const std::string&)>>(on_partial_text);

        SFSpeechRecognitionTask* task =
            [impl_->recognizer recognitionTaskWithRequest:request
                                            resultHandler:^(SFSpeechRecognitionResult* result, NSError* error) {
                                              if (result != nil) {
                                                  NSString* text = result.bestTranscription.formattedString;
                                                  if (text != nil && text.length > 0) {
                                                      const std::string text_utf8([text UTF8String]);
                                                      {
                                                          std::lock_guard<std::mutex> lock(state->mutex);
                                                          state->best_text = text_utf8;
                                                          if (!state->has_speech) {
                                                              state->has_speech = true;
                                                              state->speech_started_at = std::chrono::steady_clock::now();
                                                          }
                                                      }
                                                      if (*partial_callback) {
                                                          (*partial_callback)(text_utf8);
                                                      }
                                                  }
                                                  if (result.isFinal) {
                                                      std::lock_guard<std::mutex> lock(state->mutex);
                                                      state->done = true;
                                                  }
                                              }
                                              if (error != nil) {
                                                  const std::string err = [[error localizedDescription] UTF8String];
                                                  std::lock_guard<std::mutex> lock(state->mutex);
                                                  state->task_error = err;
                                                  state->done = true;
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
                         bufferSize:2048
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
        bool timed_out = false;
        bool phrase_timed_out = false;

        while (true) {
            bool done = false;
            bool has_speech = false;
            std::chrono::steady_clock::time_point speech_started_at{};
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                done = state->done;
                has_speech = state->has_speech;
                speech_started_at = state->speech_started_at;
            }
            if (done) {
                break;
            }

            if (tick) {
                tick();
            }

            [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.02]];

            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                timed_out = true;
                break;
            }
            if (has_speech && phrase_time_limit_seconds > 0.0) {
                const auto phrase_deadline =
                    speech_started_at + std::chrono::milliseconds(static_cast<int>(phrase_time_limit_seconds * 1000.0));
                if (now >= phrase_deadline) {
                    phrase_timed_out = true;
                    break;
                }
            }
        }

        [engine stop];
        [input_node removeTapOnBus:0];
        [request endAudio];
        [task cancel];

        std::string best_text;
        std::string task_error;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            best_text = state->best_text;
            task_error = state->task_error;
        }

        if (!task_error.empty() && best_text.empty()) {
            impl_->error = task_error;
            return std::nullopt;
        }

        if (best_text.empty()) {
            if (timed_out) {
                impl_->error = "No speech recognized before listen timeout.";
            } else if (phrase_timed_out) {
                impl_->error = "No usable speech recognized before phrase timeout.";
            } else {
                impl_->error = "No speech recognized.";
            }
            return std::nullopt;
        }

        std::string text = trim(best_text);
        if (text.empty()) {
            return std::nullopt;
        }
        return text;
    } @catch (NSException* ex) {
        impl_->error = std::string("Speech listen exception: ") + [[ex reason] UTF8String];
        return std::nullopt;
    }
}

}  // namespace alice
