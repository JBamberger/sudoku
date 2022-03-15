

/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.jbamberger.sudoku4android;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.NativeActivity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.PopupWindow;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

import timber.log.Timber;

public class NativeSudokuActivity extends NativeActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {
    private static final int PERMISSION_REQUEST_CODE_CAMERA = 1;

    static {
        System.loadLibrary("nativeui");
    }

    private final String DBG_TAG = "NDK-CAMERA-BASIC";
    volatile NativeSudokuActivity _savedInstance;
    PopupWindow _popupWindow;
    ImageButton _takePhoto;
    CameraSeekBar _exposure, _sensitivity;
    long[] _initParams;

    native static void notifyCameraPermission(boolean granted);

    native static void takePhoto();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Timber.i("OnCreate()");
        // new initialization here... request for permission
        _savedInstance = this;

        setImmersiveSticky();
        View decorView = getWindow().getDecorView();
        decorView.setOnSystemUiVisibilityChangeListener(visibility -> setImmersiveSticky());
    }

    // get current rotation method
    int getRotationDegree() {
        return 90 * ((WindowManager) (getSystemService(WINDOW_SERVICE)))
                .getDefaultDisplay()
                .getRotation();
    }

    @Override
    protected void onResume() {
        super.onResume();
        setImmersiveSticky();
    }

    void setImmersiveSticky() {
        View decorView = getWindow().getDecorView();
        decorView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    @Override
    protected void onPause() {
        if (_popupWindow != null && _popupWindow.isShowing()) {
            _popupWindow.dismiss();
            _popupWindow = null;
        }
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    public void requestCamera() {
        if (!CameraUtils.isCamera2Device(this)) {
            Log.e(DBG_TAG, "Found legacy camera Device, this sample needs camera2 device");
            return;
        }
        String[] accessPermissions = new String[]{
                Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        boolean needRequire = false;
        for (String access : accessPermissions) {
            int curPermission = ActivityCompat.checkSelfPermission(this, access);
            if (curPermission != PackageManager.PERMISSION_GRANTED) {
                needRequire = true;
                break;
            }
        }
        if (needRequire) {
            ActivityCompat.requestPermissions(
                    this, accessPermissions, PERMISSION_REQUEST_CODE_CAMERA);
            return;
        }
        notifyCameraPermission(true);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        /*
         * if any permission failed, the sample could not play
         */
        if (PERMISSION_REQUEST_CODE_CAMERA != requestCode) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }

        if (grantResults.length == 2) {
            notifyCameraPermission(grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                    grantResults[1] == PackageManager.PERMISSION_GRANTED);
        }
    }

    /**
     * params[] exposure and sensitivity init values in (min, max, curVa) tuple
     * 0: exposure min
     * 1: exposure max
     * 2: exposure val
     * 3: sensitivity min
     * 4: sensitivity max
     * 5: sensitivity val
     */
    @SuppressLint("InflateParams")
    public void updateUI(final long[] params) {
        // make our own copy
        _initParams = new long[params.length];
        System.arraycopy(params, 0, _initParams, 0, params.length);

        runOnUiThread(() -> {
            try {
                buildUI();

            } catch (WindowManager.BadTokenException e) {
                // UI error out, ignore and continue
                Log.e(DBG_TAG, "UI Exception Happened: " + e.getMessage());
            }
        });
    }

    private void buildUI() {
        if (_popupWindow != null) {
            _popupWindow.dismiss();
        }

        View popupView = ((LayoutInflater) getBaseContext()
                .getSystemService(Context.LAYOUT_INFLATER_SERVICE))
                .inflate(R.layout.widgets, null);

        _popupWindow = new PopupWindow(
                popupView,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.WRAP_CONTENT);

        RelativeLayout mainLayout = new RelativeLayout(_savedInstance);
        ViewGroup.MarginLayoutParams layoutParams = new ViewGroup.MarginLayoutParams(-1, -1);
        layoutParams.setMargins(0, 0, 0, 0);
        _savedInstance.setContentView(mainLayout, layoutParams);

        // Show our UI over NativeActivity window
        _popupWindow.showAtLocation(mainLayout, Gravity.BOTTOM | Gravity.START, 0, 0);
        _popupWindow.update();

        _takePhoto = (ImageButton) popupView.findViewById(R.id.takePhoto);
        _takePhoto.setOnClickListener(v -> takePhoto());
        _takePhoto.setEnabled(true);
        popupView.findViewById(R.id.exposureLabel).setEnabled(true);
        popupView.findViewById(R.id.sensitivityLabel).setEnabled(true);

        SeekBar exposureSeekBar = (SeekBar) popupView.findViewById(R.id.exposure_seekbar);
        _exposure = new CameraSeekBar(exposureSeekBar,
                (TextView) popupView.findViewById(R.id.exposureVal),
                _initParams[0], _initParams[1], _initParams[2]);
        exposureSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                _exposure.updateProgress(progress);
                onExposureChanged(_exposure.getAbsProgress());
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        SeekBar isoSeekBar = ((SeekBar) popupView.findViewById(R.id.sensitivity_seekbar));
        _sensitivity = new CameraSeekBar(isoSeekBar,
                (TextView) popupView.findViewById(R.id.sensitivityVal),
                _initParams[3], _initParams[4], _initParams[5]);
        isoSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                _sensitivity.updateProgress(progress);
                onSensitivityChanged(_sensitivity.getAbsProgress());
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });
    }

    /**
     * Called from Native side to notify that a photo is taken
     */
    public void onPhotoTaken(String fileName) {
        final String name = fileName;
        runOnUiThread(() -> Toast.makeText(
                getApplicationContext(), "Photo saved to " + name, Toast.LENGTH_SHORT).show());
    }

    native void onExposureChanged(long exposure);

    native void onSensitivityChanged(long sensitivity);
}

