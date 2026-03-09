#import <AppKit/AppKit.h>
#import <SceneKit/SceneKit.h>

#include "alice/ui.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

@interface AliceFaceView : NSView
@property(nonatomic, copy) NSString* stateName;
@property(nonatomic) CGFloat phase;
@property(nonatomic) CGFloat targetX;
@property(nonatomic) CGFloat targetY;
@property(nonatomic) BOOL faceLocked;
@property(nonatomic) NSInteger faceCount;
@property(nonatomic) CGFloat smoothX;
@property(nonatomic) CGFloat smoothY;

@property(nonatomic, strong) SCNView* sceneView;
@property(nonatomic, strong) SCNNode* rigNode;
@property(nonatomic, strong) SCNNode* headNode;
@property(nonatomic, strong) SCNNode* dataShellNode;
@property(nonatomic, strong) SCNNode* haloNode;
@property(nonatomic, strong) SCNNode* leftEyeNode;
@property(nonatomic, strong) SCNNode* rightEyeNode;
@property(nonatomic, strong) SCNNode* leftIrisNode;
@property(nonatomic, strong) SCNNode* rightIrisNode;
@property(nonatomic, strong) SCNNode* leftPupilNode;
@property(nonatomic, strong) SCNNode* rightPupilNode;
@property(nonatomic, strong) SCNNode* leftBrowNode;
@property(nonatomic, strong) SCNNode* rightBrowNode;
@property(nonatomic, strong) SCNNode* noseNode;
@property(nonatomic, strong) SCNNode* mouthNode;
@property(nonatomic, strong) NSArray<SCNParticleSystem*>* dissolveParticleSystems;

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
        [self buildScene];
    }
    return self;
}

- (void)layout {
    [super layout];
    if (self.sceneView != nil) {
        self.sceneView.frame = self.bounds;
    }
}

- (SCNMaterial*)materialWithDiffuse:(NSColor*)diffuse
                            specular:(NSColor*)specular
                            emission:(NSColor*)emission
                           roughness:(CGFloat)roughness
                           metalness:(CGFloat)metalness {
    SCNMaterial* material = [[SCNMaterial alloc] init];
    material.lightingModelName = SCNLightingModelPhysicallyBased;
    material.diffuse.contents = diffuse;
    material.specular.contents = specular;
    material.emission.contents = emission;
    material.roughness.contents = @(roughness);
    material.metalness.contents = @(metalness);
    material.fresnelExponent = 1.2;
    return material;
}

- (NSImage*)binaryDigitsTextureWithSize:(NSSize)size {
    NSImage* image = [[NSImage alloc] initWithSize:size];
    [image lockFocus];

    [[NSColor colorWithCalibratedWhite:0.0 alpha:1.0] setFill];
    NSRectFill(NSMakeRect(0.0, 0.0, size.width, size.height));

    NSDictionary* attrs = @{
        NSFontAttributeName : [NSFont fontWithName:@"Menlo-Bold" size:13.0] ?: [NSFont monospacedDigitSystemFontOfSize:13.0 weight:NSFontWeightMedium],
        NSForegroundColorAttributeName : [NSColor colorWithRed:0.28 green:0.94 blue:1.0 alpha:0.9],
    };

    const CGFloat stepX = 11.0;
    const CGFloat stepY = 15.0;
    const NSInteger cols = static_cast<NSInteger>(size.width / stepX) + 1;
    const NSInteger rows = static_cast<NSInteger>(size.height / stepY) + 1;

    for (NSInteger row = 0; row < rows; ++row) {
        for (NSInteger col = 0; col < cols; ++col) {
            const NSInteger pattern = (row * 7 + col * 11 + ((row + col) % 3)) % 2;
            NSString* digit = pattern == 0 ? @"0" : @"1";
            const CGFloat jitter = (row % 3 == 0 ? 1.2 : -0.8);
            const NSPoint point = NSMakePoint(col * stepX + jitter, row * stepY);
            [digit drawAtPoint:point withAttributes:attrs];
        }
    }

    [image unlockFocus];
    return image;
}

- (NSImage*)particleDigitImage {
    NSImage* image = [[NSImage alloc] initWithSize:NSMakeSize(28.0, 28.0)];
    [image lockFocus];
    [[NSColor clearColor] setFill];
    NSRectFill(NSMakeRect(0.0, 0.0, 28.0, 28.0));
    NSDictionary* attrs = @{
        NSFontAttributeName : [NSFont fontWithName:@"Menlo-Bold" size:20.0] ?: [NSFont monospacedDigitSystemFontOfSize:20.0 weight:NSFontWeightBold],
        NSForegroundColorAttributeName : [NSColor colorWithRed:0.30 green:0.96 blue:1.0 alpha:1.0],
    };
    [@"0" drawAtPoint:NSMakePoint(5.0, 3.0) withAttributes:attrs];
    [image unlockFocus];
    return image;
}

- (SCNParticleSystem*)buildDissolveSystemWithBirthRate:(CGFloat)birthRate velocity:(CGFloat)velocity {
    SCNParticleSystem* system = [SCNParticleSystem particleSystem];
    system.birthRate = birthRate;
    system.loops = YES;
    system.birthLocation = SCNParticleBirthLocationSurface;
    system.emitterShape = [SCNSphere sphereWithRadius:0.19];
    system.particleLifeSpan = 1.55;
    system.particleLifeSpanVariation = 0.55;
    system.particleVelocity = velocity;
    system.particleVelocityVariation = velocity * 0.55;
    system.spreadingAngle = 18.0;
    system.acceleration = SCNVector3Make(0.58, 0.04, 0.0);
    system.particleSize = 0.055;
    system.particleSizeVariation = 0.03;
    system.blendMode = SCNParticleBlendModeAdditive;
    system.particleColor = [NSColor colorWithRed:0.22 green:0.90 blue:1.0 alpha:0.92];
    system.particleColorVariation = SCNVector4Make(0.05, 0.08, 0.10, 0.25);
    system.particleIntensity = 1.0;
    system.particleImage = [self particleDigitImage];
    system.isAffectedByGravity = NO;
    system.isLightingEnabled = NO;
    return system;
}

- (NSColor*)stateColor {
    NSString* state = self.stateName ?: @"idle";
    if ([state isEqualToString:@"listening"]) {
        return [NSColor colorWithRed:0.20 green:0.90 blue:1.0 alpha:1.0];
    }
    if ([state isEqualToString:@"thinking"]) {
        return [NSColor colorWithRed:0.16 green:0.82 blue:0.97 alpha:1.0];
    }
    if ([state isEqualToString:@"speaking"]) {
        return [NSColor colorWithRed:0.34 green:0.96 blue:1.0 alpha:1.0];
    }
    if ([state isEqualToString:@"error"]) {
        return [NSColor colorWithRed:1.0 green:0.45 blue:0.45 alpha:1.0];
    }
    return [NSColor colorWithRed:0.18 green:0.80 blue:0.95 alpha:1.0];
}

- (void)buildScene {
    self.sceneView = [[SCNView alloc] initWithFrame:self.bounds];
    self.sceneView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.sceneView.backgroundColor = [NSColor colorWithRed:0.02 green:0.05 blue:0.09 alpha:1.0];
    self.sceneView.allowsCameraControl = NO;
    self.sceneView.playing = YES;
    self.sceneView.rendersContinuously = YES;

    SCNScene* scene = [SCNScene scene];
    self.sceneView.scene = scene;

    SCNNode* cameraNode = [SCNNode node];
    cameraNode.camera = [SCNCamera camera];
    cameraNode.camera.fieldOfView = 38.0;
    cameraNode.position = SCNVector3Make(0.0, 0.06, 8.2);
    [scene.rootNode addChildNode:cameraNode];

    SCNNode* ambient = [SCNNode node];
    ambient.light = [SCNLight light];
    ambient.light.type = SCNLightTypeAmbient;
    ambient.light.color = [NSColor colorWithRed:0.40 green:0.52 blue:0.70 alpha:1.0];
    [scene.rootNode addChildNode:ambient];

    SCNNode* key = [SCNNode node];
    key.light = [SCNLight light];
    key.light.type = SCNLightTypeOmni;
    key.light.intensity = 1200;
    key.position = SCNVector3Make(3.0, 4.2, 7.0);
    [scene.rootNode addChildNode:key];

    SCNNode* fill = [SCNNode node];
    fill.light = [SCNLight light];
    fill.light.type = SCNLightTypeOmni;
    fill.light.intensity = 560;
    fill.position = SCNVector3Make(-4.4, -1.2, 4.8);
    [scene.rootNode addChildNode:fill];

    SCNNode* rim = [SCNNode node];
    rim.light = [SCNLight light];
    rim.light.type = SCNLightTypeOmni;
    rim.light.intensity = 780;
    rim.position = SCNVector3Make(0.0, 0.6, -5.4);
    [scene.rootNode addChildNode:rim];

    self.rigNode = [SCNNode node];
    [scene.rootNode addChildNode:self.rigNode];

    SCNTorus* haloGeom = [SCNTorus torusWithRingRadius:2.42 pipeRadius:0.055];
    haloGeom.firstMaterial = [self materialWithDiffuse:[NSColor colorWithRed:0.18 green:0.42 blue:0.63 alpha:0.26]
                                              specular:[NSColor colorWithWhite:1.0 alpha:0.4]
                                              emission:[[self stateColor] colorWithAlphaComponent:0.72]
                                             roughness:0.25
                                             metalness:0.35];
    self.haloNode = [SCNNode nodeWithGeometry:haloGeom];
    self.haloNode.eulerAngles = SCNVector3Make(static_cast<float>(M_PI_2), 0.0, 0.0);
    self.haloNode.position = SCNVector3Make(0.0, 0.02, -0.18);
    [self.rigNode addChildNode:self.haloNode];

    SCNSphere* headGeom = [SCNSphere sphereWithRadius:1.90];
    headGeom.segmentCount = 88;
    headGeom.firstMaterial = [self materialWithDiffuse:[NSColor colorWithRed:0.06 green:0.26 blue:0.43 alpha:1.0]
                                              specular:[NSColor colorWithRed:0.62 green:0.84 blue:1.0 alpha:1.0]
                                              emission:[[self stateColor] colorWithAlphaComponent:0.18]
                                             roughness:0.33
                                             metalness:0.09];
    self.headNode = [SCNNode nodeWithGeometry:headGeom];
    self.headNode.position = SCNVector3Make(0.0, 0.06, 0.0);
    self.headNode.scale = SCNVector3Make(0.92, 1.15, 0.82);
    [self.rigNode addChildNode:self.headNode];

    SCNSphere* jawGeom = [SCNSphere sphereWithRadius:1.20];
    jawGeom.segmentCount = 72;
    jawGeom.firstMaterial = headGeom.firstMaterial.copy;
    SCNNode* jawNode = [SCNNode nodeWithGeometry:jawGeom];
    jawNode.position = SCNVector3Make(0.0, -1.08, 0.22);
    jawNode.scale = SCNVector3Make(0.82, 0.62, 0.70);
    [self.headNode addChildNode:jawNode];

    SCNSphere* eyeGeom = [SCNSphere sphereWithRadius:0.34];
    eyeGeom.segmentCount = 48;
    eyeGeom.firstMaterial = [self materialWithDiffuse:[NSColor colorWithRed:0.92 green:0.97 blue:1.0 alpha:1.0]
                                             specular:[NSColor colorWithRed:0.90 green:0.96 blue:1.0 alpha:1.0]
                                             emission:[NSColor colorWithRed:0.08 green:0.13 blue:0.20 alpha:0.0]
                                            roughness:0.26
                                            metalness:0.02];

    self.leftEyeNode = [SCNNode nodeWithGeometry:eyeGeom.copy];
    self.rightEyeNode = [SCNNode nodeWithGeometry:eyeGeom.copy];
    self.leftEyeNode.position = SCNVector3Make(-0.58, 0.42, 1.43);
    self.rightEyeNode.position = SCNVector3Make(0.58, 0.42, 1.43);
    [self.headNode addChildNode:self.leftEyeNode];
    [self.headNode addChildNode:self.rightEyeNode];

    SCNSphere* irisGeom = [SCNSphere sphereWithRadius:0.125];
    irisGeom.segmentCount = 36;
    irisGeom.firstMaterial = [self materialWithDiffuse:[NSColor colorWithRed:0.18 green:0.80 blue:0.98 alpha:1.0]
                                              specular:[NSColor colorWithRed:0.86 green:0.97 blue:1.0 alpha:1.0]
                                              emission:[NSColor colorWithRed:0.10 green:0.52 blue:0.68 alpha:0.45]
                                             roughness:0.22
                                             metalness:0.18];

    self.leftIrisNode = [SCNNode nodeWithGeometry:irisGeom.copy];
    self.rightIrisNode = [SCNNode nodeWithGeometry:irisGeom.copy];
    self.leftIrisNode.position = SCNVector3Make(0.0, 0.0, 0.23);
    self.rightIrisNode.position = SCNVector3Make(0.0, 0.0, 0.23);
    [self.leftEyeNode addChildNode:self.leftIrisNode];
    [self.rightEyeNode addChildNode:self.rightIrisNode];

    SCNSphere* pupilGeom = [SCNSphere sphereWithRadius:0.054];
    pupilGeom.segmentCount = 28;
    pupilGeom.firstMaterial = [self materialWithDiffuse:[NSColor colorWithRed:0.02 green:0.07 blue:0.11 alpha:1.0]
                                               specular:[NSColor colorWithRed:0.34 green:0.48 blue:0.62 alpha:1.0]
                                               emission:[NSColor colorWithRed:0.0 green:0.0 blue:0.0 alpha:0.0]
                                              roughness:0.25
                                              metalness:0.02];

    self.leftPupilNode = [SCNNode nodeWithGeometry:pupilGeom.copy];
    self.rightPupilNode = [SCNNode nodeWithGeometry:pupilGeom.copy];
    self.leftPupilNode.position = SCNVector3Make(0.0, 0.0, 0.14);
    self.rightPupilNode.position = SCNVector3Make(0.0, 0.0, 0.14);
    [self.leftIrisNode addChildNode:self.leftPupilNode];
    [self.rightIrisNode addChildNode:self.rightPupilNode];

    SCNCapsule* browGeom = [SCNCapsule capsuleWithCapRadius:0.026 height:0.52];
    browGeom.firstMaterial = [self materialWithDiffuse:[NSColor colorWithRed:0.28 green:0.82 blue:0.97 alpha:1.0]
                                              specular:[NSColor colorWithRed:0.70 green:0.94 blue:1.0 alpha:1.0]
                                              emission:[NSColor colorWithRed:0.16 green:0.54 blue:0.75 alpha:0.45]
                                             roughness:0.40
                                             metalness:0.30];

    self.leftBrowNode = [SCNNode nodeWithGeometry:browGeom.copy];
    self.rightBrowNode = [SCNNode nodeWithGeometry:browGeom.copy];
    self.leftBrowNode.position = SCNVector3Make(-0.58, 0.83, 1.57);
    self.rightBrowNode.position = SCNVector3Make(0.58, 0.83, 1.57);
    self.leftBrowNode.eulerAngles = SCNVector3Make(0.0, 0.0, 0.18);
    self.rightBrowNode.eulerAngles = SCNVector3Make(0.0, 0.0, -0.18);
    [self.headNode addChildNode:self.leftBrowNode];
    [self.headNode addChildNode:self.rightBrowNode];

    SCNCapsule* noseGeom = [SCNCapsule capsuleWithCapRadius:0.09 height:0.54];
    noseGeom.firstMaterial = [self materialWithDiffuse:[NSColor colorWithRed:0.14 green:0.42 blue:0.60 alpha:1.0]
                                              specular:[NSColor colorWithRed:0.64 green:0.86 blue:1.0 alpha:1.0]
                                              emission:[NSColor colorWithRed:0.10 green:0.30 blue:0.44 alpha:0.16]
                                             roughness:0.36
                                             metalness:0.10];
    self.noseNode = [SCNNode nodeWithGeometry:noseGeom];
    self.noseNode.position = SCNVector3Make(0.0, -0.06, 1.62);
    self.noseNode.eulerAngles = SCNVector3Make(0.0, 0.0, 0.0);
    [self.headNode addChildNode:self.noseNode];

    SCNBox* mouthGeom = [SCNBox boxWithWidth:0.68 height:0.10 length:0.08 chamferRadius:0.05];
    mouthGeom.firstMaterial = [self materialWithDiffuse:[NSColor colorWithRed:0.24 green:0.76 blue:0.93 alpha:1.0]
                                               specular:[NSColor colorWithRed:0.88 green:0.98 blue:1.0 alpha:1.0]
                                               emission:[NSColor colorWithRed:0.08 green:0.36 blue:0.50 alpha:0.36]
                                              roughness:0.28
                                              metalness:0.12];
    self.mouthNode = [SCNNode nodeWithGeometry:mouthGeom];
    self.mouthNode.position = SCNVector3Make(0.0, -0.86, 1.53);
    [self.headNode addChildNode:self.mouthNode];

    [self addSubview:self.sceneView];
}

- (void)tick {
    self.phase += 0.052;
    NSString* state = self.stateName ?: @"idle";

    CGFloat tx = self.targetX;
    CGFloat ty = self.targetY;
    if (!self.faceLocked) {
        const CGFloat sway = [state isEqualToString:@"thinking"] ? 0.13 : 0.22;
        tx = std::sin(self.phase * 0.82) * sway;
        ty = std::cos(self.phase * 0.57) * 0.14 + std::sin(self.phase * 0.33) * 0.05;
    }
    if ([state isEqualToString:@"speaking"]) {
        ty += std::sin(self.phase * 2.5) * 0.06;
    }

    self.smoothX += (tx - self.smoothX) * 0.20;
    self.smoothY += (ty - self.smoothY) * 0.20;

    const CGFloat breath = std::sin(self.phase * 0.44) * 0.045;
    self.rigNode.position = SCNVector3Make(0.0, breath, 0.0);

    const CGFloat headYaw = std::clamp(self.smoothX * 0.24, -0.34, 0.34);
    const CGFloat headPitch = std::clamp(-self.smoothY * 0.16, -0.16, 0.16);
    self.headNode.eulerAngles = SCNVector3Make(headPitch, headYaw, 0.0);

    const CGFloat blinkCycle = std::fmod(self.phase, 17.0);
    CGFloat blink = 0.0;
    if (blinkCycle > 15.8) {
        const CGFloat t = std::clamp((blinkCycle - 15.8) / 1.2, 0.0, 1.0);
        blink = std::sin(t * static_cast<CGFloat>(M_PI));
    }
    if ([state isEqualToString:@"speaking"]) {
        blink = std::max(blink, static_cast<CGFloat>(std::max(0.0, std::sin(self.phase * 1.9)) * 0.07));
    }
    const CGFloat eyeOpen = std::clamp(0.74 - blink * 0.62, 0.14, 0.82);
    self.leftEyeNode.scale = SCNVector3Make(1.0, eyeOpen, 0.92);
    self.rightEyeNode.scale = SCNVector3Make(1.0, eyeOpen, 0.92);

    const CGFloat pupilX = std::clamp(self.smoothX * 0.06, -0.06, 0.06);
    const CGFloat pupilY = std::clamp(-self.smoothY * 0.045, -0.045, 0.045);
    self.leftIrisNode.position = SCNVector3Make(pupilX, pupilY, 0.23);
    self.rightIrisNode.position = SCNVector3Make(pupilX, pupilY, 0.23);

    CGFloat mouthOpen = 0.48;
    if ([state isEqualToString:@"speaking"]) {
        mouthOpen = 0.62 + std::fabs(std::sin(self.phase * 3.2)) * 0.62;
    } else if ([state isEqualToString:@"thinking"]) {
        mouthOpen = 0.42;
    } else if ([state isEqualToString:@"listening"]) {
        mouthOpen = 0.46;
    }
    self.mouthNode.scale = SCNVector3Make(1.0, mouthOpen, 1.0);

    const CGFloat browLift = [state isEqualToString:@"thinking"] ? 0.08 : ([state isEqualToString:@"listening"] ? 0.05 : 0.03);
    const CGFloat browSway = std::sin(self.phase * 0.72) * 0.026;
    self.leftBrowNode.position = SCNVector3Make(-0.58, 0.83 + browLift + browSway, 1.57);
    self.rightBrowNode.position = SCNVector3Make(0.58, 0.83 + browLift - browSway, 1.57);

    NSColor* accent = [self stateColor];
    self.haloNode.geometry.firstMaterial.emission.contents = [accent colorWithAlphaComponent:0.72];
    self.headNode.geometry.firstMaterial.emission.contents = [accent colorWithAlphaComponent:0.16];
    self.mouthNode.geometry.firstMaterial.emission.contents = [accent colorWithAlphaComponent:0.30];

    [self.sceneView setNeedsDisplay:YES];
}

@end

@interface AliceWindowController : NSObject <NSWindowDelegate>
@property(nonatomic, strong) NSWindow* window;
@property(nonatomic, strong) AliceFaceView* faceView;
@property(nonatomic, strong) NSTextField* statusLabel;
@property(nonatomic) BOOL alive;
- (void)build;
@end

@implementation AliceWindowController

- (void)build {
    self.alive = YES;
    NSRect frame = NSMakeRect(220, 120, 760, 860);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable)
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    [self.window setTitle:@"Alice Interface"];
    [self.window setDelegate:self];

    NSView* root = [self.window contentView];

    self.faceView = [[AliceFaceView alloc] initWithFrame:NSMakeRect(24, 68, 712, 764)];
    [root addSubview:self.faceView];

    self.statusLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(24, 24, 712, 26)];
    [self.statusLabel setBezeled:NO];
    [self.statusLabel setEditable:NO];
    [self.statusLabel setDrawsBackground:NO];
    [self.statusLabel setTextColor:[NSColor colorWithRed:0.67 green:0.86 blue:1.0 alpha:1.0]];
    [self.statusLabel setAlignment:NSTextAlignmentCenter];
    [self.statusLabel setStringValue:@"Online"];
    [root addSubview:self.statusLabel];

    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)windowShouldClose:(id)sender {
    (void)sender;
    self.alive = NO;
    return YES;
}

@end

namespace alice {

struct AliceUI::Impl {
    bool is_running = false;
    std::string pending_state = "idle";
    std::string pending_status = "Online";
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

void AliceUI::add_message(const std::string&, const std::string&) {}

void AliceUI::set_face_target(float x, float y, bool found, int face_count) {
    impl_->pending_face_x = x;
    impl_->pending_face_y = y;
    impl_->pending_face_found = found;
    impl_->pending_face_count = std::max(0, face_count);
}

}  // namespace alice
