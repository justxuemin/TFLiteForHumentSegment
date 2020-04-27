package com.coocoo.tflite.activity;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;

import com.coocoo.tflite.tf.Segment;

import java.io.IOException;

public class SegmentActivitiy extends CameraActivity {

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private Segment segment;

    private Bitmap rgbFrameBitmap = null;

    private Integer sensorOrientation;

    private long lastProcessingTimeMs;

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    protected void onPreviewSizeChosen(Size size, int rotation) {
        recreateSegmenter(getModel(), getDevice(), getNumThreads());
        if (segment == null) {
            Log.e(TAG, "No segment on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();
        sensorOrientation = rotation - getScreenOrientation();

        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
    }

    private int getNumThreads() {
        return 4;
    }

    private Segment.Device getDevice() {
        return Segment.Device.GPU;
    }

    private Segment.Model getModel() {
        return Segment.Model.SHISHUAI;
    }

    private void recreateSegmenter(Segment.Model model, Segment.Device device, int numThreads) {
        if (segment != null) {
            Log.d(TAG, "Clossing segment");
            segment.close();
            segment = null;
        }
        try {
            Log.d(TAG, String.format("Creating segment (model=%s, device=%s, numThreads=%d)", model, device, numThreads));
            segment = Segment.create(this, model, device, numThreads);
        } catch (IOException e) {
            Log.e(TAG, "Failed to create segment.", e);
            return;
        }
    }


    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final int cropSize = Math.min(previewWidth, previewHeight);
        runInBackground(new Runnable() {
            @Override
            public void run() {
                if (segment != null) {
                    final long startTime = SystemClock.uptimeMillis();
                    segment.segmentImag(rgbFrameBitmap, sensorOrientation);
                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                    showInference(lastProcessingTimeMs + "ms");
                }
                readyForNextImage();
            }
        });
    }

}
