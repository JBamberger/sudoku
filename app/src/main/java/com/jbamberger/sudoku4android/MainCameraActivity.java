package com.jbamberger.sudoku4android;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import android.os.Bundle;
import android.view.WindowManager;

import java.util.Collections;
import java.util.List;

import timber.log.Timber;

public class MainCameraActivity extends CameraActivity implements CvCameraViewListener2 {
    private Mat inputFrameRgba;
    private Mat inputFrameGray;
    private CameraBridgeViewBase cvCameraView;

    private final BaseLoaderCallback cvLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Timber.i("OpenCV loaded successfully");

                // Load native library after(!) OpenCV initialization
                System.loadLibrary("mixed_sample");

                cvCameraView.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    public MainCameraActivity() {
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);

        cvCameraView = findViewById(R.id.activity_camera_surface_view);
        cvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        cvCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cvCameraView != null)
            cvCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Timber.d("Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, cvLoaderCallback);
        } else {
            Timber.d("OpenCV library found inside package. Using it!");
            cvLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(cvCameraView);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cvCameraView != null)
            cvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        String classifierPath = Utils.assetFileToLocal(this, "digit_classifier_ts.onnx");
        Timber.d("Model path: %s", classifierPath);
        init(classifierPath);

        inputFrameRgba = new Mat(height, width, CvType.CV_8UC4);
        inputFrameGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        inputFrameRgba.release();
        inputFrameGray.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        inputFrameRgba = inputFrame.rgba();
        inputFrameGray = inputFrame.gray();
        FindFeatures(inputFrameGray.getNativeObjAddr(), inputFrameRgba.getNativeObjAddr());
        return inputFrameRgba;
    }

    public native void init(String modelPath);

    public native void FindFeatures(long matAddrGr, long matAddrRgba);
}
