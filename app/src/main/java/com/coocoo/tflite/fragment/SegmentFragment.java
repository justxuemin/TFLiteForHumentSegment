package com.coocoo.tflite.fragment;

import android.app.Activity;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES31;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.text.SpannableStringBuilder;
import android.text.TextUtils;
import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.legacy.app.FragmentCompat;

import com.coocoo.tflite.R;
import com.coocoo.tflite.TFConstants;
import com.coocoo.tflite.tf.Segment;
import com.coocoo.tflite.tf.SegmentFromShiShuai;
import com.coocoo.tflite.widget.AutoFitTextureView;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import static android.opengl.EGL14.EGL_NO_CONTEXT;
import static android.opengl.GLES20.GL_BUFFER_SIZE;
import static android.opengl.GLES30.GL_STREAM_COPY;
import static android.opengl.GLES31.GL_COMPUTE_SHADER;
import static android.opengl.GLES31.GL_SHADER_STORAGE_BUFFER;

public class SegmentFragment extends Fragment implements TFConstants {

    public static SegmentFragment newInstance() {
        return new SegmentFragment();
    }

    private static final String HANDLE_THREAD_NAME = "CameraBackground";

    private AutoFitTextureView textureView;

    private HandlerThread backgroundThread;

    private Handler backgroundHandler;

    private Object lock = new Object();

    private boolean checkedPermissions = false;

    private static final int PERMISSIONS_REQUEST_CODE = 1;

    /** Max preview width that is guaranteed by Camera2 API */
    private static final int MAX_PREVIEW_WIDTH = 1920;

    /** Max preview height that is guaranteed by Camera2 API */
    private static final int MAX_PREVIEW_HEIGHT = 1080;

    private Semaphore cameraOpenCloseLock = new Semaphore(1);

    private CameraCaptureSession.CaptureCallback captureCallback =
            new CameraCaptureSession.CaptureCallback() {

                @Override
                public void onCaptureProgressed(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull CaptureResult partialResult) {}

                @Override
                public void onCaptureCompleted(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull TotalCaptureResult result) {}
            };

    private TextureView.SurfaceTextureListener surfaceTextureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int width, int height) {
            openCamera(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {

        }
    };

    private final CameraDevice.StateCallback stateCallback =
            new CameraDevice.StateCallback() {

                @Override
                public void onOpened(@NonNull CameraDevice currentCameraDevice) {
                    // This method is called when the camera is opened.  We start camera preview here.
                    cameraOpenCloseLock.release();
                    cameraDevice = currentCameraDevice;
                    createCameraPreviewSession();
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice currentCameraDevice) {
                    cameraOpenCloseLock.release();
                    currentCameraDevice.close();
                    cameraDevice = null;
                }

                @Override
                public void onError(@NonNull CameraDevice currentCameraDevice, int error) {
                    cameraOpenCloseLock.release();
                    currentCameraDevice.close();
                    cameraDevice = null;
                    Activity activity = getActivity();
                    if (null != activity) {
                        activity.finish();
                    }
                }
            };
    private ImageReader imageReader;
    private Size previewSize;
    private String cameraId;
    private CameraDevice cameraDevice;
    private CaptureRequest.Builder previewRequestBuilder;
    private SurfaceTexture camSurfTex;
    private CameraCaptureSession captureSession;
    private CaptureRequest previewRequest;
    private boolean runSegment = false;
    private Segment segment;
    private EGLDisplay eglDisplay;
    private EGLContext eglContext;
    private EGLDisplay gpuDisplay;
    private EGLContext gpuContext;
    private EGLConfig gpuConfig;
    private EGLSurface gpuSurface;
    private EGLConfig eglConfig;
    private EGLSurface eglSurface;
    private int camTexId;
    private int camSsboId;
    private int camToSsboProgId;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_camera, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        textureView = (AutoFitTextureView) view.findViewById(R.id.texture);
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        startBackgroundThread();
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();

        if (textureView.isAvailable()) {
            openCamera(textureView.getWidth(), textureView.getHeight());
        } else {
            textureView.setSurfaceTextureListener(surfaceTextureListener);
        }
    }

    private void openCamera(int width, int height) {
        if (!checkedPermissions && !allPermissionsGranted()) {
            FragmentCompat.requestPermissions(this, getRequiredPermissions(), PERMISSIONS_REQUEST_CODE);
            return;
        } else {
            checkedPermissions = true;
        }
        setUpCameraOutputs(width, height);
        configureTransform(width, height);
        Activity activity = getActivity();
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            manager.openCamera(cameraId, stateCallback, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Failed to open Camera", e);
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
        }
    }

    private void configureTransform(int viewWidth, int viewHeight) {
        Activity activity = getActivity();
        if (null == textureView || null == previewSize || null == activity) {
            return;
        }
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, previewSize.getHeight(), previewSize.getWidth());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            float scale =
                    Math.max(
                            (float) viewHeight / previewSize.getHeight(),
                            (float) viewWidth / previewSize.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        } else if (Surface.ROTATION_180 == rotation) {
            matrix.postRotate(180, centerX, centerY);
        }
        textureView.setTransform(matrix);
    }

    private void createCameraPreviewSession() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;

            // We configure the size of default buffer to be the size of camera preiew we want.
            texture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());

            // This is the output Surface we need to start preview.
            Surface surface = new Surface(texture);

            // We set up a CaptureRequest.Builder with the output Surface.
            previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            previewRequestBuilder.addTarget(surface);



            // create SSBO
            initSsbo();

            // create a new surface texture to get the camera frame by using GLES texture name
            int preview_cx = previewSize.getWidth();
            int preview_cy = previewSize.getHeight();

            camSurfTex = new SurfaceTexture(camTexId);
            camSurfTex.setDefaultBufferSize(preview_cx, preview_cy);

            // create a camera surface target
            Surface camSurf = new Surface (camSurfTex);
            SurfaceTexture.OnFrameAvailableListener camFrameListener = new SurfaceTexture.OnFrameAvailableListener() {
                @Override
                public void onFrameAvailable(SurfaceTexture surfaceTexture) {
                    //camSurfTex.updateTexImage();

                    synchronized(lock) {
                        runSegment = true;
                    }
                    //if (classifier != null && camSsboId != 0 && classifier.getOutputSsboId() != 0) {
                    //  long startTime = SystemClock.uptimeMillis();
                    //  classifier.displayOutputSsboToTextureView();
                    //  long endTime = SystemClock.uptimeMillis();
                    //  Log.d(TAG, "Time cost to display output SSBO: " + Long.toString(endTime - startTime));
                    //  Log.d(TAG, "------------------------------------------------------------");
                    //}
                }
            };

            camSurfTex.setOnFrameAvailableListener(camFrameListener);

            previewRequestBuilder.addTarget(camSurf);


            // Here, we create a CameraCaptureSession for camera preview.
            cameraDevice.createCaptureSession(
                    //Arrays.asList(surface),
                    Arrays.asList(surface, camSurf),
                    //Arrays.asList(camSurf),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                            // The camera is already closed
                            if (null == cameraDevice) {
                                return;
                            }

                            // When the session is ready, we start displaying the preview.
                            captureSession = cameraCaptureSession;
                            try {
                                // Auto focus should be continuous for camera preview.
                                previewRequestBuilder.set(
                                        CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                                // Finally, we start displaying the camera preview.
                                previewRequest = previewRequestBuilder.build();
                                captureSession.setRepeatingRequest(
                                        previewRequest, captureCallback, backgroundHandler);
                            } catch (CameraAccessException e) {
                                Log.e(TAG, "Failed to set up config to capture Camera", e);
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                            //showToast("Failed");
                            Log.e(TAG, "cameraCaptureSession failed");
                        }
                    },
                    null);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Failed to preview Camera", e);
        }
    }

    private static class CompareSizesByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum(
                    (long) lhs.getWidth() * lhs.getHeight() - (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    private void setUpCameraOutputs(int width, int height) {
        Activity activity = getActivity();
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing != CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map == null) {
                    continue;
                }

                // // For still image captures, we use the largest available size.
                Size largest =
                        Collections.max(
                                Arrays.asList(map.getOutputSizes(ImageFormat.JPEG)), new CompareSizesByArea());
                imageReader =
                        ImageReader.newInstance(
                                largest.getWidth(), largest.getHeight(), ImageFormat.JPEG, /*maxImages*/ 2);

                // Find out if we need to swap dimension to get the preview size relative to sensor
                // coordinate.
                int displayRotation = activity.getWindowManager().getDefaultDisplay().getRotation();
                // noinspection ConstantConditions
                /* Orientation of the camera sensor */
                int sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
                boolean swappedDimensions = false;
                switch (displayRotation) {
                    case Surface.ROTATION_0:
                    case Surface.ROTATION_180:
                        if (sensorOrientation == 90 || sensorOrientation == 270) {
                            swappedDimensions = true;
                        }
                        break;
                    case Surface.ROTATION_90:
                    case Surface.ROTATION_270:
                        if (sensorOrientation == 0 || sensorOrientation == 180) {
                            swappedDimensions = true;
                        }
                        break;
                    default:
                        Log.e(TAG, "Display rotation is invalid: " + displayRotation);
                }

                Point displaySize = new Point();
                activity.getWindowManager().getDefaultDisplay().getSize(displaySize);
                int rotatedPreviewWidth = width;
                int rotatedPreviewHeight = height;
                int maxPreviewWidth = displaySize.x;
                int maxPreviewHeight = displaySize.y;

                if (swappedDimensions) {
                    rotatedPreviewWidth = height;
                    rotatedPreviewHeight = width;
                    maxPreviewWidth = displaySize.y;
                    maxPreviewHeight = displaySize.x;
                }

                if (maxPreviewWidth > MAX_PREVIEW_WIDTH) {
                    maxPreviewWidth = MAX_PREVIEW_WIDTH;
                }

                if (maxPreviewHeight > MAX_PREVIEW_HEIGHT) {
                    maxPreviewHeight = MAX_PREVIEW_HEIGHT;
                }

                previewSize = chooseOptimalSize(
                                map.getOutputSizes(SurfaceTexture.class),
                                rotatedPreviewWidth,
                                rotatedPreviewHeight);

                // We fit the aspect ratio of TextureView to the size of preview we picked.
                int orientation = getResources().getConfiguration().orientation;
                if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
                    textureView.setAspectRatio(previewSize.getWidth(), previewSize.getHeight());
                } else {
                    textureView.setAspectRatio(previewSize.getHeight(), previewSize.getWidth());
                }

                this.cameraId = cameraId;
                return;
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, "Failed to access Camera", e);
        } catch (NullPointerException e) {
            // Currently an NPE is thrown when the Camera2API is used but not supported on the
            // device this code runs.
            //CameraConnectionFragment.ErrorDialog.newInstance(getString(R.string.camera_error))
                    //.show(getChildFragmentManager(), FRAGMENT_DIALOG);
            Log.e(TAG, "Failed to access Camera", e);
        }
    }

    /**
     * 提供camera提供的size选择
     * @param choices
     * @param width
     * @param height
     * @return
     */
    private static final int MINIMUM_PREVIEW_SIZE = 320;
    protected static Size chooseOptimalSize(final Size[] choices, final int width, final int height) {
        final int minSize = Math.max(Math.min(width, height), MINIMUM_PREVIEW_SIZE);
        final Size desiredSize = new Size(width, height);

        // Collect the supported resolutions that are at least as big as the preview Surface
        boolean exactSizeFound = false;
        final List<Size> bigEnough = new ArrayList<Size>();
        final List<Size> tooSmall = new ArrayList<Size>();
        for (final Size option : choices) {
            if (option.equals(desiredSize)) {
                // Set the size but don't return yet so that remaining sizes will still be logged.
                exactSizeFound = true;
            }

            if (option.getHeight() >= minSize && option.getWidth() >= minSize) {
                bigEnough.add(option);
            } else {
                tooSmall.add(option);
            }
        }

        Log.i(TAG, "Desired size: " + desiredSize + ", min size: " + minSize + "x" + minSize);
        Log.i(TAG, "Valid preview sizes: [" + TextUtils.join(", ", bigEnough) + "]");
        Log.i(TAG, "Rejected preview sizes: [" + TextUtils.join(", ", tooSmall) + "]");

        if (exactSizeFound) {
            Log.i(TAG, "Exact size match found.");
            return desiredSize;
        }

        // Pick the smallest of those, assuming we found any
        if (bigEnough.size() > 0) {
            final Size chosenSize = Collections.min(bigEnough, new CameraConnectionFragment.CompareSizesByArea());
            Log.i(TAG, "Chosen size: " + chosenSize.getWidth() + "x" + chosenSize.getHeight());
            return chosenSize;
        } else {
            Log.e(TAG, "Couldn't find any suitable preview size");
            return choices[0];
        }
    }

    private static Size chooseOptimalSize(
            Size[] choices,
            int textureViewWidth,
            int textureViewHeight,
            int maxWidth,
            int maxHeight,
            Size aspectRatio) {

        // Collect the supported resolutions that are at least as big as the preview Surface
        List<Size> bigEnough = new ArrayList<>();
        // Collect the supported resolutions that are smaller than the preview Surface
        List<Size> notBigEnough = new ArrayList<>();
        int w = aspectRatio.getWidth();
        int h = aspectRatio.getHeight();
        for (Size option : choices) {
            if (option.getWidth() <= maxWidth
                    && option.getHeight() <= maxHeight
                    && option.getHeight() == option.getWidth() * h / w) {
                if (option.getWidth() >= textureViewWidth && option.getHeight() >= textureViewHeight) {
                    bigEnough.add(option);
                } else {
                    notBigEnough.add(option);
                }
            }
        }

        // Pick the smallest of those big enough. If there is no one big enough, pick the
        // largest of those not big enough.
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else if (notBigEnough.size() > 0) {
            return Collections.max(notBigEnough, new CompareSizesByArea());
        } else {
            Log.e(TAG, "Couldn't find any suitable preview size");
            return choices[0];
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            if (ContextCompat.checkSelfPermission(getActivity(), permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private String[] getRequiredPermissions() {
        Activity activity = getActivity();
        try {
            PackageInfo info = activity.getPackageManager().getPackageInfo(activity.getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }

    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();

    }

    @Override
    public void onDestroy() {
        if (segment != null) {
            segment.close();
        }
        super.onDestroy();
    }

    private void closeCamera() {
        try {
            cameraOpenCloseLock.acquire();
            if (null != captureSession) {
                captureSession.close();
                captureSession = null;
            }
            if (null != cameraDevice) {
                cameraDevice.close();
                cameraDevice = null;
            }
            if (null != imageReader) {
                imageReader.close();
                imageReader = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            cameraOpenCloseLock.release();
        }
    }

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        // Start the classification train & load an initial model.
        synchronized (lock) {
            //runClassifier = true;

            // only run when there is a frame available
            runSegment = false;
        }
        backgroundHandler.post(periodicClassify);
        updateActiveModel();
    }

    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
            synchronized (lock) {
                runSegment = false;
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted when stopping background thread", e);
        }
    }

    /** Takes photos and classify them periodically. */
    private Runnable periodicClassify =
            new Runnable() {
                @Override
                public void run() {
                    synchronized (lock) {
                        if (runSegment) {
                            if (segment != null) {
//                                if (segment.getInputSsboId() == 0) {
//                                    classifyFrame();
//                                } else {
                                    // wait until next frame in the SurfaceTexture frame listener in onFrameAvailable()
                                    classifyFrameSSBO();
                                    runSegment = false;
//                                }
                            }
                        }
                    }
                    backgroundHandler.post(periodicClassify);
                }
            };

    private void updateActiveModel() {
        // Get UI information before delegating to background
        backgroundHandler.post(new Runnable() {
            @Override
            public void run() {

                // Disable classifier while updating
                if (segment != null) {
                    segment.close();
                    segment = null;
                }

                // Try to load model.
                try {
                    segment = new SegmentFromShiShuai(SegmentFragment.this.getActivity(), Segment.Device.GPU, 4);
                } catch (IOException e) {
                    Log.d(TAG, "Failed to load", e);
                    segment = null;
                }

                // Customize the interpreter to the type of device we want to use.
                if (segment == null) {
                    return;
                }
                segment.setNumThreads(4);

                EGLDisplay[] display = new EGLDisplay[1];
                EGLContext[] context = new EGLContext[1];
                EGLConfig[] config = new EGLConfig[1];
                EGLSurface[] surface = new EGLSurface[1];

                if (eglContext == null) {
                    return;
                }

                initGLES(context, display, config, surface, eglContext);
                gpuDisplay = display[0];
                gpuContext = context[0];
                gpuConfig = config[0];
                gpuSurface = surface[0];

                // make the gpu context current before calling useGpu(), which in turn calls
                //      modifyGraphWithDelegate()
                EGL14.eglMakeCurrent(gpuDisplay, gpuSurface, gpuSurface, gpuContext);

                // set the SSBO
                segment.setInputSsboId(camSsboId);
                segment.useGpu();

                // resumes normal egl context
                EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);

            }
        });
    }

    void initSsbo () {
        // add an off-screen surface target, so pixels can be retrieved from cam preview frames.
        int preview_cx = previewSize.getWidth();
        int preview_cy = previewSize.getHeight();

        // init GLES
        EGLDisplay[] display = new EGLDisplay[1];
        EGLContext[] context = new EGLContext[1];
        EGLConfig [] config  = new EGLConfig [1];
        EGLSurface[] surface = new EGLSurface[1];

        initGLES(context, display, config, null, EGL14.EGL_NO_CONTEXT);
        eglDisplay = display[0];
        eglContext = context[0];
        eglConfig  = config [0];
        //eglSurface = surface[0];

        // create a real on-display surface that associates GLES
        int[] eglSurfaceAttribs = {
                EGL14.EGL_NONE
        };

        // On Pixel 3 (Android P, Snapdragon 845), a real on-display surface needs to be associated to make
        //      compute shader works.
        // On LG G4 (Android M, Snapdragon 808), however, only a dummy surface is needed.
        // For Pixel 3 and LG G4, a real on-display surface works on both devices.
        SurfaceTexture outputSurf = textureView.getSurfaceTexture();
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, eglConfig, outputSurf, eglSurfaceAttribs, 0);

        // make the context current for current thread, display and surface.
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);

        // TODO: remove hard code.
        int img_cx = 224;
        int img_cy = 224;

        //==========  cam -> tex -> ssbo  ==========
        // create a texture name
        camTexId = createTextureName();

        camSsboId = createSSBO(img_cx, img_cy);
        camToSsboProgId = createShaderProgram_TexToSsbo(preview_cx, preview_cy, img_cx, img_cy);
    }

    private int createTextureName () {
        // create texture name
        int[] texIds = new int[1];
        GLES31.glGenTextures(texIds.length, texIds, 0);

        if (texIds[0] == 0) {
            throw new RuntimeException("cannot create texture name.");
        }

        int texId = texIds[0];

        return texId;
    }

    private int createSSBO (int cx, int cy) {
        int[] ssboIds = new int[1];
        GLES31.glGenBuffers(ssboIds.length, ssboIds, 0);

        int ssboId = ssboIds[0];
        GLES31.glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboId);

        if (ssboId == 0) {
            throw new RuntimeException("cannot create SSBO.");
        }

        ByteBuffer ssboData = null;

        int PIXEL_SIZE = 3;
        int FLOAT_BYTE_SIZE = 4;
        int ssboSize = cx * cy * PIXEL_SIZE * FLOAT_BYTE_SIZE;

        //+++++ for debug purpose
        boolean DEBUG = false;
        if (DEBUG) {
            // byte buffer to initialize ssbo buffer
            ssboData = ByteBuffer.allocateDirect(ssboSize);
            ssboData.order(ByteOrder.nativeOrder());

            // create a left-to-right gradient
            for (int y=0; y<cy; y++) {
                for (int x=0; x<cx; x++) {
                    float r = 0.0f;
                    float g = ((float)(x))/((float)cx);
                    float b = 0.0f;

                    ssboData.putFloat(r);
                    ssboData.putFloat(g);
                    ssboData.putFloat(b);
                }
            }

            ssboData.rewind();
        }
        //----- for debug purpose

        GLES31.glBufferData(GL_SHADER_STORAGE_BUFFER, ssboSize, ssboData, GL_STREAM_COPY);

        int ssboSizeCreated [] = new int[1];
        GLES31.glGetBufferParameteriv (GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, ssboSizeCreated, 0);
        if (ssboSizeCreated[0] != ssboSize) {
            throw new RuntimeException("cannot create SSBO with needed size.");
        }

        GLES31.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);  // unbind output ssbo

        return ssboId;
    }

    // image size should be always smaller than camera size
    //  cam_cx >= img_cy
    //  cam_cy >= img_cy
    private int createShaderProgram_TexToSsbo (int cam_cx, int cam_cy, int img_cx, int img_cy) {
        // create ssbo --> texture shader program
        String shaderCode =
                "   #version 310 es\n" +
                        "   #extension GL_OES_EGL_image_external_essl3: enable\n" +
                        "   precision mediump float;\n" + // need to specify 'mediump' for float
                        //"   layout(local_size_x = 16, local_size_y = 16) in;\n" +
                        "   layout(local_size_x = 8, local_size_y = 8) in;\n" +
                        //"   layout(binding = 0) uniform sampler2D in_data; \n" +
                        "   layout(binding = 0) uniform samplerExternalOES in_data; \n" +
                        "   layout(std430) buffer;\n" +
                        "   layout(binding = 1) buffer Input { float elements[]; } out_data;\n" +
                        "   void main() {\n" +
                        "     ivec2 gid = ivec2(gl_GlobalInvocationID.xy);\n" +
                        "     if (gid.x >= " + img_cx + " || gid.y >= " + img_cy + ") return;\n" +
                        "     vec2 uv = vec2(gl_GlobalInvocationID.xy) / " + img_cx + ".0;\n" +
                        "     vec4 pixel = texture (in_data, uv);\n" +
                        "     int idx = 3 * (gid.y * " + img_cx + " + gid.x);\n" +
                        //"     if (gid.x >= 120) pixel.x = 1.0;\n" + // DEBUG...
                        "     out_data.elements[idx + 0] = pixel.x;\n" +
                        "     out_data.elements[idx + 1] = pixel.y;\n" +
                        "     out_data.elements[idx + 2] = pixel.z;\n" +
                        "   }";

        int shader = GLES31.glCreateShader(GL_COMPUTE_SHADER);
        GLES31.glShaderSource(shader, shaderCode);
        GLES31.glCompileShader(shader);

        int[] compiled = new int [1];
        GLES31.glGetShaderiv(shader, GLES31.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            // shader compilation failed
            String log = "shader - compilation error: " + GLES31.glGetShaderInfoLog(shader);

            Log.i(TAG, log);
            throw new RuntimeException(log);
        }

        int progId = GLES31.glCreateProgram();
        if (progId == 0) {
            String log = "shader - cannot create program";

            Log.i(TAG, log);
            throw new RuntimeException(log);
        }

        GLES31.glAttachShader(progId, shader);
        GLES31.glLinkProgram (progId);

        int[] linked = new int[1];
        GLES31.glGetProgramiv(progId, GLES31.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            String log = "shader - link error - log: " + GLES31.glGetProgramInfoLog(progId);

            Log.i(TAG, log);
            throw new RuntimeException(log);
        }

        return progId;
    }

    private void initGLES (EGLContext[] context, EGLDisplay[] display, EGLConfig[] config, EGLSurface[] surface, EGLContext sharedCntx){

        EGLDisplay disp = null;
        EGLContext cntx = null;
        EGLConfig  cfig = null;
        EGLSurface surf = null;

        // egl init
        disp = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (disp == null) {
            throw new RuntimeException("unable to get EGL14 display");
        }

        int[] vers = new int[2];
        if (!EGL14.eglInitialize(disp, vers, 0, vers, 1)) {
            throw new RuntimeException("unable to initialize EGL14 display");
        }

        int[] configAttr = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT | EGLExt.EGL_OPENGL_ES3_BIT_KHR,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
        };

        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfig = new int[1];
        EGL14.eglChooseConfig(disp, configAttr, 0,
                configs, 0, 1, numConfig, 0);
        if (numConfig[0] == 0) {
            throw new RuntimeException("unable to choose config for EGL14 display");
        }
        cfig = configs[0];

        int[] ctxAttrib = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
        };

        // create egl context
        cntx = EGL14.eglCreateContext(disp, cfig, sharedCntx, ctxAttrib, 0);  // needs a shared context for a shared ssbo.
        if (cntx.equals(EGL_NO_CONTEXT)) {
            throw new RuntimeException("unable to create EGL14 context");
        }
        //Log.i(TAG, "Camera2BasicFragment - created egl context");


        context[0] = cntx;
        display[0] = disp;
        config [0] = cfig;

        if (surface != null) {
            // create a dummy surface as no on-screen drawing is needed
            SurfaceTexture dummySurf = new SurfaceTexture(true);

            // the surface is to display the output to the screen
            int[] dummySurfAttrib = {
                    EGL14.EGL_NONE
            };
            surf = EGL14.eglCreateWindowSurface(disp, cfig, dummySurf, dummySurfAttrib, 0);

            surface[0] = surf;
        }
    }

    private void classifyFrameSSBO() {
        if (segment == null || getActivity() == null || cameraDevice == null) {
            // It's important to not call showToast every frame, or else the app will starve and
            // hang. updateActiveModel() already puts a error message up with showToast.
            // showToast("Uninitialized Classifier or invalid context.");
            return;
        }

        SpannableStringBuilder textToShow = new SpannableStringBuilder();
        //Bitmap bitmap = textureView.getBitmap(classifier.getImageSizeX(), classifier.getImageSizeY());
        //classifier.classifyFrame(bitmap, textToShow);
        //bitmap.recycle();


        // update the texture image to get the camera frame
        camSurfTex.updateTexImage();

        // set gles context
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);

        // copy the surface to texture
        long copy_t0 = SystemClock.uptimeMillis();
        copyCamTexToSsbo();
        long copy_t1 = SystemClock.uptimeMillis();

        // classify the frame in the SSBO
        EGL14.eglMakeCurrent(gpuDisplay, gpuSurface, gpuSurface, gpuContext);
        segment.segmentFrameSSBO(copy_t1 - copy_t0);

        // resumes the normal egl context
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);

        //showToast(textToShow);
    }

    void copyCamTexToSsbo () {
        // bind camera input texture to GL_TEXTURE_EXTERNAL_OES.

        // only copy up to classifier image size
        int img_cx = segment.getImageSizeX();//previewSize.getWidth();
        int img_cy = segment.getImageSizeY();//previewSize.getHeight();

        int FLOAT_BYTE_SIZE = 4;
        int camSsboSize = img_cx * img_cy * 3 * FLOAT_BYTE_SIZE;  // input ssbo to tflite gpu delegate has 3 channels

        // updateTexImage() binds the texture to GL_TEXTURE_EXTERNAL_OES
        GLES31.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, camTexId);
        GLES31.glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, camSsboId, 0, camSsboSize);

        GLES31.glUseProgram(camToSsboProgId);
        //GLES31.glDispatchCompute(img_cx / 16, img_cy / 16, 1);  // these are work group sizes
        GLES31.glDispatchCompute(img_cx / 8, img_cy / 8, 1);  // smaller work group sizes for lower end GPU.

        GLES31.glMemoryBarrier(GLES31.GL_SHADER_STORAGE_BARRIER_BIT);

        GLES31.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);  // unbind
        GLES31.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);  // unbind
    }

}
