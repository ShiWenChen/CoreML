//
//  ViewController.m
//  CoreML
//
//  Created by ShiWen on 2017/8/31.
//  Copyright © 2017年 ShiWen. All rights reserved.
//

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <Vision/Vision.h>
#import <Vision/Vision.h>
#import <AVFoundation/AVFoundation.h>
#import "MobileNet.h"


@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>

@property (nonatomic,strong) AVCaptureSession *captureSession;
@property (nonatomic,strong) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic,strong) dispatch_queue_t captureQueue;
@property (nonatomic,strong) VNRequest *visionCoreMLRequest;
@property (nonatomic,strong) VNRequest *visionFaceLandmarkRequest;
@property (nonatomic,strong) AVCaptureInput *currentInput;
@property (nonatomic,strong) AVCaptureOutput *currentOutput;

@property (nonatomic,strong) VNSequenceRequestHandler *sequenceRequestHandler;
@property (nonatomic,strong)AVCaptureDevice *captureDevice;
@property (nonatomic,strong) UILabel *lbObject;


@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setupVisionRequests];
    [self setCamera];
}
-(void)setCamera{
    NSError *error;
    self.currentInput = [[AVCaptureDeviceInput alloc] initWithDevice:self.captureDevice error:&error];
    //    添加
    if ([self.captureSession canAddInput:self.currentInput]) {
        [self.captureSession addInput:self.currentInput];
    }else{
        NSLog(@"创建失败");
    }
    
    
    //   创建输出队列
    dispatch_queue_t cameraQueue = dispatch_queue_create("cameraQueue", NULL);
    AVCaptureVideoDataOutput *videoOutPutData = [[AVCaptureVideoDataOutput alloc]init];
    self.currentOutput = videoOutPutData;
    [videoOutPutData setSampleBufferDelegate:self queue:cameraQueue];
    videoOutPutData.alwaysDiscardsLateVideoFrames = NO;
    //    视频格式设置
    [videoOutPutData setVideoSettings:@{(id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)}];
    [videoOutPutData connectionWithMediaType:AVMediaTypeVideo];
    
    //    添加输出并显示预览
    if ([self.captureSession canAddOutput:videoOutPutData]) {
        [self.captureSession addOutput:videoOutPutData];
    }
    
    AVCaptureConnection *connection = [videoOutPutData connectionWithMediaType:AVMediaTypeVideo];
    //    设置视频方向
    [connection setVideoOrientation:AVCaptureVideoOrientationPortraitUpsideDown];
    self.previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.captureSession];
    
    self.previewLayer.videoGravity = AVLayerVideoGravityResize;
    [self.view.layer addSublayer:self.previewLayer];
    self.view.backgroundColor = [UIColor redColor];
    //    提交配置
    [self.captureSession commitConfiguration];
    //    开始采集
    [self.captureSession startRunning];
    
    [self.view addSubview:self.lbObject];
    
}

-(void)viewDidLayoutSubviews{
    self.previewLayer.frame = self.view.bounds;
}
-(AVCaptureSession *)captureSession{
    if (!_captureSession) {
        _captureSession = [[AVCaptureSession alloc]init];
        //        设置清晰度
        _captureSession.sessionPreset = AVCaptureSessionPreset1280x720;
    }
    return _captureSession;
}
//设备
-(AVCaptureDevice *)captureDevice{
    if (!_captureDevice) {
        _captureDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        //        后摄像头
        _captureDevice = [self getCameraDeviceWithPostion:AVCaptureDevicePositionBack];
    }
    return _captureDevice;
}
-(UILabel *)lbObject{
    if (!_lbObject) {
        _lbObject = [[UILabel alloc] initWithFrame:CGRectMake(10, 20, 355, 100)];
        _lbObject.numberOfLines = 0;
        _lbObject.textAlignment = NSTextAlignmentCenter;
        _lbObject.backgroundColor = [UIColor colorWithRed:206/255.0 green:206/255.0 blue:206/255.0 alpha:0.5];
        _lbObject.font = [UIFont boldSystemFontOfSize:16];
        
    }
    return _lbObject;
}
-(AVCaptureDevice*)getCameraDeviceWithPostion:(AVCaptureDevicePosition ) postion{
    NSArray *cameras = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *camera in cameras) {
        if ([camera position] == postion) {
            return camera;
        }
    }
    return nil;
}


/**
 获取视频流
 */
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection{
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!pixelBuffer) {
        return;
    }
    AVCaptureInput *input = connection.inputPorts.firstObject.input;
    if (input != self.currentInput) {
        return;
    }
    NSMutableDictionary *requestOptions = [NSMutableDictionary dictionary];
    CFTypeRef cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil);
    if (cameraIntrinsicData) {
        requestOptions[VNImageOptionCameraIntrinsics] = (__bridge id _Nullable)(cameraIntrinsicData);
    }
    if (!self.sequenceRequestHandler) {
        self.sequenceRequestHandler = [[VNSequenceRequestHandler alloc]init];
    }
    NSError *error = [[NSError alloc] init];
    BOOL isScuuess = [self.sequenceRequestHandler performRequests:@[self.visionCoreMLRequest] onCVPixelBuffer:pixelBuffer error:&error];
    if (!isScuuess) {
        NSLog(@"失败%@",error);
    }
    
}
-(void)setupVisionRequests
{
    MobileNet *mobilenetModel = [[MobileNet alloc] init];
    VNCoreMLModel *visionModel = [VNCoreMLModel modelForMLModel:mobilenetModel.model error:nil];
    VNCoreMLRequest *classificationRequest = [[VNCoreMLRequest alloc] initWithModel:visionModel completionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
        if (error) {
            NSLog(@"Failed:%@",error);
        }
        NSArray *observations = request.results;
        if (!observations.count) {
            return NSLog(@"无数据");
        }
        VNClassificationObservation *observation = nil;
        for (VNClassificationObservation *ob in observations) {
            if (![ob isKindOfClass:[VNClassificationObservation class]]) {
                continue;
            }
            if (!observations) {
                observation = ob;
                continue;
            }
            if (observation.confidence < ob.confidence) {
                observation = ob;
            }
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            NSMutableString *text = [NSMutableString string];
            for (NSString *str in [observation.identifier componentsSeparatedByString:@","]) {
                NSString *strName = [str stringByAppendingString:@"->"];
                [text appendString:strName];
            }
            self.lbObject.text = [NSString stringWithFormat:@"我找到了：%@",text];
        });
    }];
    self.visionCoreMLRequest = classificationRequest;
    
////    人脸识别         暂未启用 可自行添加
//    VNDetectFaceLandmarksRequest *faceRequest = [[VNDetectFaceLandmarksRequest alloc]initWithCompletionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
//        void (^finish)(VNFaceObservation *,NSString *) = ^(VNFaceObservation *ob,NSString *text){
//            dispatch_async(dispatch_get_main_queue(), ^{
//                NSLog(@"%@",ob);
//                self.lbObject.text = text;
//            });
//
//        };
//        if (error) {
//            return finish(nil,error.localizedDescription);
//        }
//        NSArray *observations = request.results;
//        if (!observations.count) {
//            return finish(nil,@"未识别人脸");
//        }
//        VNFaceObservation *observation = nil;
//        for (VNFaceObservation *ob in observations) {
//            if (![ob isKindOfClass:[VNFaceObservation class]]) {
//                continue;
//            }
//            if (!observation) {
//                observation = ob;
//                continue;
//            }
//            if (observation.confidence<ob.confidence) {
//                observation = ob;
//            }
//        }
//        finish(observation,[NSString stringWithFormat:@"(%.0f%%)",observation.confidence *100]);
//
//    }];
//
//    self.visionFaceLandmarkRequest = faceRequest;

    
    
    
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
