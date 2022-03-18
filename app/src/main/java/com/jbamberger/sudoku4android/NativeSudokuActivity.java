

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
import android.app.NativeActivity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.PopupWindow;
import android.widget.RelativeLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

import timber.log.Timber;

public class NativeSudokuActivity extends NativeActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {
    private static final int PERMISSION_REQUEST_CODE_CAMERA = 1;

    static {
        System.loadLibrary("sudoku_android_app");
    }

    volatile NativeSudokuActivity _savedInstance;
    PopupWindow _popupWindow;
    ImageButton _takePhoto;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        _savedInstance = this;

        setImmersiveSticky();
        View decorView = getWindow().getDecorView();
        decorView.setOnSystemUiVisibilityChangeListener(visibility -> setImmersiveSticky());
    }


    @Override
    protected void onResume() {
        super.onResume();
        setImmersiveSticky();
    }


    @Override
    protected void onPause() {
        if (_popupWindow != null && _popupWindow.isShowing()) {
            _popupWindow.dismiss();
            _popupWindow = null;
        }
        super.onPause();
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

    public void requestCamera() {
        if (!CameraUtils.isCamera2Device(this)) {
            Timber.e("Found legacy camera Device, this sample needs camera2 device");
            return;
        }

        String[] permissions = new String[]{
                Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        boolean permMissing = false;
        for (String permission : permissions) {
            int permissionResult = ActivityCompat.checkSelfPermission(this, permission);
            permMissing = permMissing || (permissionResult != PackageManager.PERMISSION_GRANTED);
        }

        if (permMissing) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE_CAMERA);
            return;
        }

        notifyCameraPermission(true);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (PERMISSION_REQUEST_CODE_CAMERA != requestCode) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }

        if (grantResults.length == 2) {
            notifyCameraPermission(grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                    grantResults[1] == PackageManager.PERMISSION_GRANTED);
        }
    }

    public void updateUI() {
        runOnUiThread(() -> {
            try {
                buildUI();
            } catch (WindowManager.BadTokenException e) {
                // UI error out, ignore and continue
                Timber.e("UI Exception Happened: %s", e.getMessage());
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

        RelativeLayout mainLayout = new RelativeLayout(this);
        ViewGroup.MarginLayoutParams layoutParams = new ViewGroup.MarginLayoutParams(-1, -1);
        layoutParams.setMargins(0, 0, 0, 0);
        setContentView(mainLayout, layoutParams);

        // Show our UI over NativeActivity window
        _popupWindow.showAtLocation(mainLayout, Gravity.BOTTOM | Gravity.START, 0, 0);
        _popupWindow.update();

        _takePhoto = popupView.findViewById(R.id.takePhoto);
        _takePhoto.setOnClickListener(v -> takePhoto());
        _takePhoto.setEnabled(true);
    }

    public void onPhotoTaken(String fileName) {
        final String name = fileName;
        runOnUiThread(() -> Toast.makeText(
                getApplicationContext(), "Photo saved to " + name, Toast.LENGTH_SHORT).show());
    }

    public int getRotationDegree() {
        return 90 * ((WindowManager) (getSystemService(WINDOW_SERVICE)))
                .getDefaultDisplay()
                .getRotation();
    }

    public String getModelPath() {
        String classifierPath = Utils.assetFileToLocal(this, "digit_classifier_ts.onnx");
        return classifierPath;
    }

    native static void notifyCameraPermission(boolean granted);

    native static void takePhoto();
}

