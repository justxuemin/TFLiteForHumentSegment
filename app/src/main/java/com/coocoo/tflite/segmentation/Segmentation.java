package com.coocoo.tflite.segmentation;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;
import android.util.Log;

import com.coocoo.tflite.TFConstants;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class Segmentation implements TFConstants {

    private final Context mContext;
    private final ByteBuffer segmentationMasks;
    private int numThreads;

    private int inputSsboId;

    private GpuDelegate gpuDelegate;

    private Interpreter.Options tfliteOptions = new Interpreter.Options();

    private Interpreter tflite;

    private MappedByteBuffer tfliteModel;

    private static final int imageSize = 257;
    private static final int NUM_CLASSES = 21;

    public Segmentation(Context context) {
        mContext = context;
        tfliteModel = loadModelFile(context);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        segmentationMasks = ByteBuffer.allocateDirect(256*256*2*4);
        segmentationMasks.order(ByteOrder.nativeOrder());
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Context context){
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = context.getAssets().openFd(getModelPath());
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            throw new RuntimeException();
        }
    }

    private String getModelPath() {
        return "shishuai.tflite";
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
        tfliteOptions.setNumThreads(numThreads);
        recreateInterpreter();
    }

    public void setInputSsboId (int ssboId) {
        inputSsboId = ssboId;
    }

    public int  getInputSsboId () {
        return inputSsboId;
    }

    public void useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = new GpuDelegate();

            // use only FP16 precision for lower memory requirement
            tfliteOptions.setAllowFp16PrecisionForFp32(true);

            //tfliteOptions.addDelegate(gpuDelegate);
            //recreateInterpreter();

            // if no input ssbo, use the option to add gpu delegate
            if (inputSsboId == 0) {
                tfliteOptions.addDelegate(gpuDelegate);
                recreateInterpreter();
            } else {
                recreateInterpreter();
            }
        }
    }

    private void recreateInterpreter() {
        if (tflite != null) {
            tflite.close();
            tflite = new Interpreter(tfliteModel, tfliteOptions);

            // binding input ssbo
            if (inputSsboId != 0) {
                Tensor inputTensor = tflite.getInputTensor(0);
                gpuDelegate.bindGlBufferToTensor(inputTensor, inputSsboId);

                tflite.modifyGraphWithDelegate(gpuDelegate);
            }
        }
    }

    public void close() {
        tflite.close();
        tflite = null;
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        tfliteModel = null;
    }


    public int getImageSizeX() {
        return 256;
    }

    public int getImageSizeY() {
        return 256;
    }

    // classify a frame using SSBO
    public void classifyFrameSSBO(SpannableStringBuilder builder, long copyTime) {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            builder.append(new SpannableString("Uninitialized Classifier."));
        }

        // add measurement for pixel copy
        //long pixelCopyStart = SystemClock.uptimeMillis();
        //convertBitmapToByteBuffer(bitmap);
        //long pixelCopyEnd = SystemClock.uptimeMillis();

        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        runInference();
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

        // Smooth the results across frames.
        convertBytebufferMaskToBitmap();

        long duration = endTime - startTime;
        SpannableString span = new SpannableString(duration + " ms" + "     copy time: " + (copyTime) + "ms");
        span.setSpan(new ForegroundColorSpan(android.graphics.Color.LTGRAY), 0, span.length(), 0);
        builder.append(span);
    }

    int[][] mSegmentBits = new int[256][256];
    private void convertBytebufferMaskToBitmap() {
        StringBuilder sb = new StringBuilder();
        Bitmap maskBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
        for (int y = 0; y < 256; y++) {
            for (int x = 0; x < 256;  x++) {
                float maxVal = 0f;
                mSegmentBits[x][y] = 0;
                for (int c = 0; c < 2; c++) {
                    float value = segmentationMasks.getFloat((y * 256 * 2 + x * 2 + c) * 4);
                    if (c == 0 /*|| value > maxVal*/) {
                        maxVal = value;
                        mSegmentBits[x][y] = value < 0 ? 0 : 255;
                    }
                    sb.append(value + "\t");
                }
//                if (mSegmentBits[x][y] == 0) {
//                    sb.append(mSegmentBits[x][y] + "\t");
//                }
                maskBitmap.setPixel(x, y, mSegmentBits[x][y] == 0? Color.RED : Color.BLUE);
            }
        }
        Log.e("tflite", "result:" + sb.toString());
    }

    private void runInference() {
        segmentationMasks.rewind();
        tflite.run(null, segmentationMasks);
    }
}
