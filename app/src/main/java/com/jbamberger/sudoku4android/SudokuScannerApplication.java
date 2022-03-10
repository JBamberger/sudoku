package com.jbamberger.sudoku4android;

import android.app.Application;

import timber.log.Timber;

public class SudokuScannerApplication extends Application {

    @Override
    public void onCreate() {
        super.onCreate();

        if (BuildConfig.DEBUG) {
            Timber.plant(new Timber.DebugTree());
        }
    }
}
