#import <AppKit/AppKit.h>

#include "alice/ui.hpp"

#include <cmath>
#include <deque>
#include <utility>
#include <vector>

@interface AliceFaceView : NSView
@property(nonatomic, copy) NSString* stateName;
@property(nonatomic) CGFloat phase;
- (void)tick;
@end

@implementation AliceFaceView

- (instancetype)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        _stateName = @"idle";
        _phase = 0.0;
        self.wantsLayer = YES;
    }
    return self;
}

- (NSColor*)ringColor {
    NSString* state = self.stateName ?: @"idle";
    if ([state isEqualToString:@"listening"]) {
        return [NSColor colorWithRed:0.43 green:0.72 blue:1.0 alpha:1.0];
    }
    if ([state isEqualToString:@"thinking"]) {
        return [NSColor colorWithRed:0.95 green:0.80 blue:0.45 alpha:1.0];
    }
    if ([state isEqualToString:@"speaking"]) {
        return [NSColor colorWithRed:0.48 green:0.90 blue:0.72 alpha:1.0];
    }
    if ([state isEqualToString:@"error"]) {
        return [NSColor colorWithRed:1.0 green:0.58 blue:0.58 alpha:1.0];
    }
    return [NSColor colorWithRed:0.32 green:0.50 blue:0.72 alpha:1.0];
}

- (void)tick {
    self.phase += 0.08;
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];

    NSRect bounds = self.bounds;
    [[NSColor colorWithRed:0.05 green:0.09 blue:0.13 alpha:1.0] setFill];
    NSRectFill(bounds);

    NSBezierPath* halo = [NSBezierPath bezierPathWithOvalInRect:NSInsetRect(bounds, 35, 25)];
    [[NSColor colorWithRed:0.10 green:0.20 blue:0.32 alpha:1.0] setStroke];
    [halo setLineWidth:2.0];
    [halo stroke];

    NSBezierPath* head = [NSBezierPath bezierPathWithOvalInRect:NSInsetRect(bounds, 90, 70)];
    [[NSColor colorWithRed:0.06 green:0.15 blue:0.23 alpha:1.0] setFill];
    [head fill];
    [[self ringColor] setStroke];
    [head setLineWidth:3.0];
    [head stroke];

    const CGFloat cx = NSMidX(bounds);
    const CGFloat cy = NSMidY(bounds) + 18;

    const CGFloat eyeW = 60;
    const CGFloat eyeH = 44;
    NSRect leftEye = NSMakeRect(cx - 90, cy + 8, eyeW, eyeH);
    NSRect rightEye = NSMakeRect(cx + 30, cy + 8, eyeW, eyeH);

    [[NSColor colorWithRed:0.07 green:0.18 blue:0.28 alpha:1.0] setFill];
    [[NSBezierPath bezierPathWithOvalInRect:leftEye] fill];
    [[NSBezierPath bezierPathWithOvalInRect:rightEye] fill];

    const CGFloat driftX = std::sin(self.phase) * 5.0;
    const CGFloat driftY = std::cos(self.phase * 0.7) * 3.0;
    NSRect leftPupil = NSMakeRect(NSMidX(leftEye) - 9 + driftX, NSMidY(leftEye) - 9 + driftY, 18, 18);
    NSRect rightPupil = NSMakeRect(NSMidX(rightEye) - 9 + driftX, NSMidY(rightEye) - 9 + driftY, 18, 18);
    [[NSColor colorWithRed:0.38 green:0.86 blue:1.0 alpha:1.0] setFill];
    [[NSBezierPath bezierPathWithOvalInRect:leftPupil] fill];
    [[NSBezierPath bezierPathWithOvalInRect:rightPupil] fill];

    CGFloat mouthOpen = 10.0;
    if ([self.stateName isEqualToString:@"speaking"]) {
        mouthOpen = 10.0 + std::fabs(std::sin(self.phase * 2.6)) * 15.0;
    } else if ([self.stateName isEqualToString:@"thinking"]) {
        mouthOpen = 8.0;
    }

    NSBezierPath* mouth = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(cx - 44, cy - 58, 88, mouthOpen)
                                                          xRadius:10
                                                          yRadius:10];
    [[NSColor colorWithRed:0.16 green:0.46 blue:0.67 alpha:1.0] setFill];
    [mouth fill];
}

@end

@interface AliceWindowController : NSObject <NSWindowDelegate>
@property(nonatomic, strong) NSWindow* window;
@property(nonatomic, strong) AliceFaceView* faceView;
@property(nonatomic, strong) NSTextField* statusLabel;
@property(nonatomic, strong) NSTextView* chatView;
@property(nonatomic) BOOL alive;
- (void)build;
@end

@implementation AliceWindowController

- (void)build {
    self.alive = YES;
    NSRect frame = NSMakeRect(200, 160, 560, 760);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable)
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    [self.window setTitle:@"Alice Interface"];
    [self.window setDelegate:self];

    NSView* root = [self.window contentView];

    self.faceView = [[AliceFaceView alloc] initWithFrame:NSMakeRect(30, 330, 500, 390)];
    [root addSubview:self.faceView];

    self.statusLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(30, 300, 500, 24)];
    [self.statusLabel setBezeled:NO];
    [self.statusLabel setEditable:NO];
    [self.statusLabel setDrawsBackground:NO];
    [self.statusLabel setTextColor:[NSColor colorWithRed:0.67 green:0.86 blue:1.0 alpha:1.0]];
    [self.statusLabel setStringValue:@"Online"];
    [root addSubview:self.statusLabel];

    NSScrollView* scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(30, 30, 500, 250)];
    [scroll setHasVerticalScroller:YES];
    [scroll setBorderType:NSNoBorder];

    self.chatView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, 500, 250)];
    [self.chatView setEditable:NO];
    [self.chatView setBackgroundColor:[NSColor colorWithRed:0.07 green:0.12 blue:0.18 alpha:1.0]];
    [self.chatView setTextColor:[NSColor colorWithRed:0.86 green:0.92 blue:1.0 alpha:1.0]];
    [self.chatView setFont:[NSFont fontWithName:@"Avenir Next" size:14.0]];
    [scroll setDocumentView:self.chatView];
    [root addSubview:scroll];

    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)windowShouldClose:(id)sender {
    self.alive = NO;
    return YES;
}

@end

namespace alice {

struct AliceUI::Impl {
    bool is_running = false;
    std::string pending_state = "idle";
    std::string pending_status = "Online";
    std::deque<std::pair<std::string, std::string>> pending_messages;
    AliceWindowController* controller = nil;
};

AliceUI::AliceUI() : impl_(std::make_unique<Impl>()) {}
AliceUI::~AliceUI() {
    stop();
}

bool AliceUI::start() {
    if (impl_->is_running) {
        return true;
    }

    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        impl_->controller = [[AliceWindowController alloc] init];
        [impl_->controller build];
        impl_->is_running = (impl_->controller != nil);
    }

    return impl_->is_running;
}

void AliceUI::pump() {
    if (!impl_->is_running || impl_->controller == nil) {
        return;
    }

    impl_->controller.faceView.stateName = [NSString stringWithUTF8String:impl_->pending_state.c_str()];
    impl_->controller.statusLabel.stringValue = [NSString stringWithUTF8String:impl_->pending_status.c_str()];

    while (!impl_->pending_messages.empty()) {
        const auto msg = impl_->pending_messages.front();
        impl_->pending_messages.pop_front();
        NSString* line = [NSString stringWithFormat:@"%s> %s\n", msg.first.c_str(), msg.second.c_str()];
        NSAttributedString* attr = [[NSAttributedString alloc] initWithString:line];
        [[impl_->controller.chatView textStorage] appendAttributedString:attr];
        [impl_->controller.chatView scrollRangeToVisible:NSMakeRange(impl_->controller.chatView.string.length, 0)];
    }

    [impl_->controller.faceView tick];

    NSEvent* event = nil;
    do {
        event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                   untilDate:[NSDate dateWithTimeIntervalSinceNow:0.0]
                                      inMode:NSDefaultRunLoopMode
                                     dequeue:YES];
        if (event != nil) {
            [NSApp sendEvent:event];
        }
    } while (event != nil);
    [NSApp updateWindows];

    if (!impl_->controller.alive || ![impl_->controller.window isVisible]) {
        impl_->is_running = false;
    }
}

void AliceUI::stop() {
    if (!impl_->is_running) {
        return;
    }
    if (impl_->controller != nil) {
        [impl_->controller.window close];
        impl_->controller = nil;
    }
    impl_->is_running = false;
}

bool AliceUI::running() const {
    return impl_->is_running;
}

void AliceUI::set_state(const std::string& state) {
    impl_->pending_state = state;
}

void AliceUI::set_status(const std::string& status) {
    impl_->pending_status = status;
}

void AliceUI::add_message(const std::string& speaker, const std::string& text) {
    impl_->pending_messages.emplace_back(speaker, text);
}

}  // namespace alice
