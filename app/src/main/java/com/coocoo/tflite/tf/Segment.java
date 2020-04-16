package com.coocoo.tflite.tf;

import android.app.Activity;
import android.graphics.Bitmap;
import android.util.Log;

import com.coocoo.tflite.TFConstants;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public abstract class Segment implements TFConstants {

    public enum Model {
        SHISHUAI
    }

    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    /** Image size along the x axis. */
    private final int imageSizeX;

    /** Image size along the y axis. */
    private final int imageSizeY;

    protected Interpreter tflite;
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /**
     * 加载的TF lite模型
     */
    private MappedByteBuffer tfliteModel;

    private TensorImage inputImageBuffer;
    private final TensorBuffer outputProbabilityBuffer;
    private final TensorProcessor probabilityProcessor;

    protected Segment(Activity activity, Device device, int numThreads) throws IOException {
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
        switch (device) {
            case NNAPI:
                //nnApiDelegate = new NnApiDelegate();
                //tfliteOptions.addDelegate(nnApiDelegate);
                break;
            case GPU:
                //gpuDelegate = new GpuDelegate();
                //tfliteOptions.addDelegate(gpuDelegate);
                break;
            case CPU:
                break;
            default:
                break;
        }
        tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        // 输入输出tensors的type 和 shape
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // 创建输入tensor
        inputImageBuffer = new TensorImage(imageDataType);

        // 创建输出tensor
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        //创建输出概率处理器
        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        Log.d(TAG, "Created a Tensorflow Lite segment.");
    }

    protected abstract String getModelPath();

    protected abstract TensorOperator getPostprocessNormalizeOp();

    protected abstract TensorOperator getPreprocessNormalizeOp();

    /**
     * 关闭解释器，模型以及释放资源
     */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        /*if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }*/
        /*if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }*/
        tfliteModel = null;
    }

    public void segmentImag(final Bitmap bitmap, int sensorOrientation) {

        inputImageBuffer = loadImage(bitmap, sensorOrientation);

        // 调用推断
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

    }

    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(numRotation))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    /** Get the image size along the x axis. */
    public int getImageSizeX() {
        return imageSizeX;
    }

    /** Get the image size along the y axis. */
    public int getImageSizeY() {
        return imageSizeY;
    }

    public static Segment create(Activity activity, Model model, Device device, int numThreads)
            throws IOException {
        if (model == Model.SHISHUAI) {
            return new SegmentFromShiShuai(activity, device, numThreads);
        } else {
            throw new UnsupportedOperationException();
        }
    }

}
