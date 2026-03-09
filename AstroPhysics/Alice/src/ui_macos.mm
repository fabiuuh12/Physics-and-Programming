#import <AppKit/AppKit.h>

#include "alice/ui.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <utility>
#include <vector>

@interface AliceFaceView : NSView
@property(nonatomic, copy) NSString* stateName;
@property(nonatomic) CGFloat phase;
@property(nonatomic) CGFloat targetX;
@property(nonatomic) CGFloat targetY;
@property(nonatomic) BOOL faceLocked;
@property(nonatomic) NSInteger faceCount;
@property(nonatomic) CGFloat smoothX;
@property(nonatomic) CGFloat smoothY;
- (void)tick;
@end

@implementation AliceFaceView

- (instancetype)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        _stateName = @"idle";
        _phase = 0.0;
        _targetX = 0.0;
        _targetY = 0.0;
        _faceLocked = NO;
        _faceCount = 0;
        _smoothX = 0.0;
        _smoothY = 0.0;
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
    self.phase += 0.06;

    CGFloat tx = self.targetX;
    CGFloat ty = self.targetY;
    if (!self.faceLocked) {
        tx = std::sin(self.phase * 0.8) * 0.22;
        ty = std::cos(self.phase * 0.55) * 0.14;
    }

    self.smoothX += (tx - self.smoothX) * 0.18;
    self.smoothY += (ty - self.smoothY) * 0.18;
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    NSRect bounds = self.bounds;

    NSGradient* bg = [[NSGradient alloc] initWithStartingColor:[NSColor colorWithRed:0.03 green:0.06 blue:0.10 alpha:1.0]
                                                   endingColor:[NSColor colorWithRed:0.08 green:0.12 blue:0.18 alpha:1.0]];
    [bg drawInRect:bounds angle:90.0];

    NSBezierPath* aura1 = [NSBezierPath bezierPathWithOvalInRect:NSInsetRect(bounds, 28, 20)];
    [[NSColor colorWithRed:0.12 green:0.26 blue:0.40 alpha:0.8] setStroke];
    [aura1 setLineWidth:2.0];
    [aura1 stroke];

    NSBezierPath* aura2 = [NSBezierPath bezierPathWithOvalInRect:NSInsetRect(bounds, 48, 40)];
    [[NSColor colorWithRed:0.10 green:0.22 blue:0.34 alpha:0.7] setStroke];
    [aura2 setLineWidth:1.4];
    [aura2 stroke];

    const CGFloat cx = NSMidX(bounds) + self.smoothX * 14.0;
    const CGFloat cy = NSMidY(bounds) + 18.0 + self.smoothY * 10.0;

    NSRect headRect = NSMakeRect(cx - 140, cy - 138, 280, 286);
    NSBezierPath* head = [NSBezierPath bezierPathWithOvalInRect:headRect];

    NSGradient* headGrad = [[NSGradient alloc] initWithColors:@[
        [NSColor colorWithRed:0.05 green:0.18 blue:0.29 alpha:1.0],
        [NSColor colorWithRed:0.07 green:0.30 blue:0.44 alpha:1.0],
        [NSColor colorWithRed:0.03 green:0.13 blue:0.20 alpha:1.0],
    ]];
    [headGrad drawInBezierPath:head angle:120.0];
    [[self ringColor] setStroke];
    [head setLineWidth:3.0];
    [head stroke];

    NSBezierPath* cheekGlow = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(cx - 112, cy - 46, 224, 188)];
    NSGradient* cheekGrad = [[NSGradient alloc] initWithStartingColor:[NSColor colorWithRed:0.53 green:0.84 blue:1.0 alpha:0.16]
                                                          endingColor:[NSColor colorWithRed:0.08 green:0.20 blue:0.29 alpha:0.00]];
    [cheekGrad drawInBezierPath:cheekGlow angle:90.0];

    const CGFloat eyeYOffset = 46;
    const CGFloat eyeDX = 58;
    NSRect leftEye = NSMakeRect(cx - eyeDX - 34, cy + eyeYOffset - 22, 68, 46);
    NSRect rightEye = NSMakeRect(cx + eyeDX - 34, cy + eyeYOffset - 22, 68, 46);

    NSGradient* scleraGrad = [[NSGradient alloc] initWithStartingColor:[NSColor colorWithRed:0.88 green:0.96 blue:1.0 alpha:0.95]
                                                           endingColor:[NSColor colorWithRed:0.52 green:0.68 blue:0.80 alpha:0.95]];
    [scleraGrad drawInBezierPath:[NSBezierPath bezierPathWithOvalInRect:leftEye] angle:90.0];
    [scleraGrad drawInBezierPath:[NSBezierPath bezierPathWithOvalInRect:rightEye] angle:90.0];

    const CGFloat pupilOffsetX = std::clamp(self.smoothX * 10.0, -9.0, 9.0);
    const CGFloat pupilOffsetY = std::clamp(self.smoothY * 7.0, -7.0, 7.0);

    NSRect leftIris = NSMakeRect(NSMidX(leftEye) - 12 + pupilOffsetX, NSMidY(leftEye) - 12 + pupilOffsetY, 24, 24);
    NSRect rightIris = NSMakeRect(NSMidX(rightEye) - 12 + pupilOffsetX, NSMidY(rightEye) - 12 + pupilOffsetY, 24, 24);
    [[NSColor colorWithRed:0.22 green:0.84 blue:0.99 alpha:1.0] setFill];
    [[NSBezierPath bezierPathWithOvalInRect:leftIris] fill];
    [[NSBezierPath bezierPathWithOvalInRect:rightIris] fill];

    NSRect leftPupil = NSInsetRect(leftIris, 7, 7);
    NSRect rightPupil = NSInsetRect(rightIris, 7, 7);
    [[NSColor colorWithRed:0.02 green:0.07 blue:0.12 alpha:1.0] setFill];
    [[NSBezierPath bezierPathWithOvalInRect:leftPupil] fill];
    [[NSBezierPath bezierPathWithOvalInRect:rightPupil] fill];

    [[NSColor colorWithRed:0.85 green:0.98 blue:1.0 alpha:0.85] setFill];
    [[NSBezierPath bezierPathWithOvalInRect:NSMakeRect(NSMinX(leftIris) + 5, NSMinY(leftIris) + 15, 5, 5)] fill];
    [[NSBezierPath bezierPathWithOvalInRect:NSMakeRect(NSMinX(rightIris) + 5, NSMinY(rightIris) + 15, 5, 5)] fill];

    NSBezierPath* nose = [NSBezierPath bezierPath];
    [nose moveToPoint:NSMakePoint(cx, cy + 28)];
    [nose lineToPoint:NSMakePoint(cx - 11, cy - 22)];
    [nose lineToPoint:NSMakePoint(cx + 11, cy - 22)];
    [nose closePath];
    [[NSColor colorWithRed:0.12 green:0.38 blue:0.56 alpha:0.86] setFill];
    [nose fill];

    CGFloat mouthOpen = 10.0;
    if ([self.stateName isEqualToString:@"speaking"]) {
        mouthOpen = 12.0 + std::fabs(std::sin(self.phase * 2.9)) * 18.0;
    } else if ([self.stateName isEqualToString:@"thinking"]) {
        mouthOpen = 8.0;
    }

    NSRect mouthOuter = NSMakeRect(cx - 52, cy - 70, 104, 24);
    NSBezierPath* upperLip = [NSBezierPath bezierPath];
    [upperLip moveToPoint:NSMakePoint(NSMinX(mouthOuter), NSMidY(mouthOuter))];
    [upperLip curveToPoint:NSMakePoint(NSMaxX(mouthOuter), NSMidY(mouthOuter))
             controlPoint1:NSMakePoint(cx - 24, NSMaxY(mouthOuter) + 6)
             controlPoint2:NSMakePoint(cx + 24, NSMaxY(mouthOuter) + 6)];
    [[NSColor colorWithRed:0.32 green:0.78 blue:0.95 alpha:1.0] setStroke];
    [upperLip setLineWidth:3.0];
    [upperLip stroke];

    NSBezierPath* mouthInner = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(cx - 38, cy - 68, 76, mouthOpen)
                                                               xRadius:10
                                                               yRadius:10];
    [[NSColor colorWithRed:0.06 green:0.20 blue:0.33 alpha:0.95] setFill];
    [mouthInner fill];

    if (self.faceLocked) {
        NSString* tag = self.faceCount > 1 ? [NSString stringWithFormat:@"Tracking %ld faces", (long)self.faceCount]
                                           : @"Tracking face";
        NSDictionary* attrs = @{ NSForegroundColorAttributeName : [NSColor colorWithRed:0.70 green:0.92 blue:1.0 alpha:0.95],
                                 NSFontAttributeName : [NSFont fontWithName:@"Avenir Next" size:12.0] };
        [tag drawAtPoint:NSMakePoint(12, bounds.size.height - 24) withAttributes:attrs];
    }
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
    float pending_face_x = 0.0f;
    float pending_face_y = 0.0f;
    bool pending_face_found = false;
    int pending_face_count = 0;
    AliceWindowController* controller = nil;
};

AliceUI::AliceUI() : impl_(std::make_unique<Impl>()) {}
AliceUI::~AliceUI() { stop(); }

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
    impl_->controller.faceView.targetX = std::clamp(static_cast<CGFloat>(impl_->pending_face_x), -1.0, 1.0);
    impl_->controller.faceView.targetY = std::clamp(static_cast<CGFloat>(impl_->pending_face_y), -1.0, 1.0);
    impl_->controller.faceView.faceLocked = impl_->pending_face_found ? YES : NO;
    impl_->controller.faceView.faceCount = impl_->pending_face_count;

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

bool AliceUI::running() const { return impl_->is_running; }

void AliceUI::set_state(const std::string& state) { impl_->pending_state = state; }

void AliceUI::set_status(const std::string& status) { impl_->pending_status = status; }

void AliceUI::add_message(const std::string& speaker, const std::string& text) {
    impl_->pending_messages.emplace_back(speaker, text);
}

void AliceUI::set_face_target(float x, float y, bool found, int face_count) {
    impl_->pending_face_x = x;
    impl_->pending_face_y = y;
    impl_->pending_face_found = found;
    impl_->pending_face_count = std::max(0, face_count);
}

}  // namespace alice
