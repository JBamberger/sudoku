package com.jbamberger.sudoku4android;

import android.content.Context;

import androidx.annotation.Nullable;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import timber.log.Timber;

class Utils {

    @Nullable
    public static String assetFileToLocal(Context context, String file) {
        var assetManager = context.getAssets();
        var outFile = new File(context.getFilesDir(), file);

        try (var source = new BufferedInputStream(assetManager.open(file));
             var target = new FileOutputStream(outFile)) {
            byte[] buf = new byte[8192];
            int length;
            while ((length = source.read(buf)) > 0) {
                target.write(buf, 0, length);
            }

            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Timber.i("Failed to upload a file");
        }
        return null;
    }
}
