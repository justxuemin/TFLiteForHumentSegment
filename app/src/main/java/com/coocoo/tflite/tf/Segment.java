package com.coocoo.tflite.tf;

import android.app.Activity;
import android.graphics.Bitmap;
import android.opengl.GLES31;
import android.util.Log;

import com.coocoo.tflite.TFConstants;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Arrays;

import static android.opengl.EGL14.eglGetCurrentContext;
import static android.opengl.GLES31.GL_SHADER_STORAGE_BUFFER;

public abstract class Segment implements TFConstants {

    private final Device mDevice;
    private int mNumThreads;
    private boolean init = false;
    private int inputSsboId;

    public int getInputSsboId() {
        return inputSsboId;
    }

    public void setInputSsboId(int camSsboId) {
        inputSsboId = camSsboId;
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

    public enum Model {
        SHISHUAI
    }

    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    /** Image size along the x axis. */
    private int imageSizeX;

    /** Image size along the y axis. */
    private int imageSizeY;

    protected Interpreter tflite;
    private Interpreter.Options tfliteOptions = new Interpreter.Options();

    private GpuDelegate gpuDelegate = null;

    /**
     * 加载的TF lite模型
     */
    private MappedByteBuffer tfliteModel;

    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;

    protected Segment(Activity activity, Device device, int numThreads) throws IOException {
        mDevice = device;
        mNumThreads = numThreads;
        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
        Log.d(TAG, "Created a Tensorflow Lite segment.");

    }

    public int[] initializeShaderBuffer(){
        android.opengl.EGLContext eglContext = eglGetCurrentContext();
        int[] id = new int[1];
        GLES31.glGenBuffers(id.length, id, 0);
        GLES31.glBindBuffer(GL_SHADER_STORAGE_BUFFER, id[0]);
        GLES31.glBufferData(GL_SHADER_STORAGE_BUFFER, 257*257*3*4, null, GLES31.GL_STREAM_COPY);

        GLES31.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);// unbind
        return id;
    }

    public void updateActiveModel() {

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
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        /*if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }*/
        tfliteModel = null;
    }

    public void segmentImag(final Bitmap bitmap, int sensorOrientation) {

        if (!init) {
            switch (mDevice) {
                case NNAPI:
                    //nnApiDelegate = new NnApiDelegate();
                    //tfliteOptions.addDelegate(nnApiDelegate);
                    break;
                case GPU:
                    gpuDelegate = new GpuDelegate();
                    tfliteOptions.addDelegate(gpuDelegate);
                    break;
                case CPU:
                    break;
                default:
                    break;
            }
            tfliteOptions.setNumThreads(mNumThreads);
            tflite = new Interpreter(tfliteModel, tfliteOptions);

            // 输入输出tensors的type 和 shape
            int imageTensorIndex = 0;
            Tensor inputTensor = tflite.getInputTensor(imageTensorIndex);
            int[] imageShape = inputTensor.shape(); // {1, height, width, 3}
            int numDimensions = inputTensor.numDimensions();
            Log.e("xuemin", "tensor in numDimensions : " + numDimensions);
            Log.e("xuemin", "tensor in shape " + Arrays.toString(imageShape));
            imageSizeY = imageShape[1];
            imageSizeX = imageShape[2];
            DataType imageDataType = inputTensor.dataType();
            int probabilityTensorIndex = 0;
            Tensor outputTensor = tflite.getOutputTensor(probabilityTensorIndex);
            int[] probabilityShape = outputTensor.shape(); // {1, NUM_CLASSES}
            DataType probabilityDataType = outputTensor.dataType();
            int numDimensions1 = outputTensor.numDimensions();
            Log.e("xuemin", "tensor out numDimensions : " + numDimensions1);
            Log.e("xuemin", "tensor out shape " + Arrays.toString(probabilityShape));
            // 创建输入tensor
            inputImageBuffer = new TensorImage(imageDataType);

            // 创建输出tensor
            outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

            //创建输出概率处理器
            probabilityProcessor = new TensorProcessor.Builder().build();
            init = true;
        }

        inputImageBuffer = loadImage(bitmap, sensorOrientation);

        Log.e("xuemin", "tensor in buffer" + Arrays.toString(inputImageBuffer.getBuffer().array()));
        // 调用推断
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

        Log.e("xuemin", "tensor out buffer" + Arrays.toString(outputProbabilityBuffer.getBuffer().array()));
        //bitmap.copyPixelsFromBuffer(outputProbabilityBuffer.getBuffer());




    }

    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        Log.e("xuemin", "tensor load image " + Arrays.toString(inputImageBuffer.getBuffer().array()));

        // Creates processor for the TensorImage.
        //int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new Rot90Op(numRotation))
                        //.add(getPreprocessNormalizeOp())
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

    public void setNumThreads(int numThreads) {
        mNumThreads = numThreads;
        tfliteOptions.setNumThreads(numThreads);
    }

}
