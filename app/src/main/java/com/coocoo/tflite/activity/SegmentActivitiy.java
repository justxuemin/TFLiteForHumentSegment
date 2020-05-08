package com.coocoo.tflite.activity;

import android.app.Activity;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Bitmap;
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
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;

import androidx.annotation.NonNull;

import com.coocoo.tflite.R;
import com.coocoo.tflite.tf.Segment;
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

public class SegmentActivitiy extends CameraActivity {

    /** Max preview width that is guaranteed by Camera2 API */
    private static final int MAX_PREVIEW_WIDTH = 1920;

    /** Max preview height that is guaranteed by Camera2 API */
    private static final int MAX_PREVIEW_HEIGHT = 1080;

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;

    private static final String HANDLE_THREAD_NAME = "CameraBackground";

    private final Object lock = new Object();
    private boolean runSegment = false;

    private AutoFitTextureView glesView;

    EGLContext eglContext = null;
    EGLDisplay eglDisplay = null;
    EGLSurface eglSurface = null;
    EGLConfig  eglConfig  = null;

    EGLDisplay gpuDisplay = null;
    EGLSurface gpuSurface = null;
    EGLContext gpuContext = null;
    EGLConfig  gpuConfig  = null;

    int camSsboId = 0;
    int camTexId = 0;

    SurfaceTexture camSurfTex = null;

    int camToSsboProgId = 0;

    private ImageReader imageReader;

    private String cameraId;

    private Size previewSize;

    private CaptureRequest.Builder previewRequestBuilder;




    /** A {@link Semaphore} to prevent the app from exiting before closing the camera. */
    private Semaphore cameraOpenCloseLock = new Semaphore(1);

    private CameraDevice cameraDevice;

    private CameraCaptureSession captureSession;

    private CaptureRequest previewRequest;

    private final TextureView.SurfaceTextureListener surfaceTextureListener =
            new TextureView.SurfaceTextureListener() {

                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
                    openCamera(width, height);
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {
                    configureTransform(width, height);
                }

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture texture) {}
            };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        glesView = findViewById(R.id.gles);
    }

    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();

        // When the screen is turned off and turned back on, the SurfaceTexture is already
        // available, and "onSurfaceTextureAvailable" will not be called. In that case, we can open
        // a camera and start preview from here (otherwise, we wait until the surface is ready in
        // the SurfaceTextureListener).
        if (glesView.isAvailable()) {
            openCamera(glesView.getWidth(), glesView.getHeight());
        } else {
            glesView.setSurfaceTextureListener(surfaceTextureListener);
        }
    }

    private void openCamera(int width, int height) {
        setUpCameraOutputs(width, height);
        configureTransform(width, height);
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
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

    /** {@link CameraDevice.StateCallback} is called when {@link CameraDevice} changes its state. */
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
                    finish();
                }
            };

    /** Creates a new {@link CameraCaptureSession} for camera preview. */
    private void createCameraPreviewSession() {
        try {
            SurfaceTexture texture = glesView.getSurfaceTexture();
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
                        }
                    },
                    null);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Failed to preview Camera", e);
        }
    }

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

    /**
     * Configures the necessary {@link android.graphics.Matrix} transformation to `textureView`. This
     * method should be called after the camera preview size is determined in setUpCameraOutputs and
     * also the size of `textureView` is fixed.
     *
     * @param viewWidth The width of `textureView`
     * @param viewHeight The height of `textureView`
     */
    private void configureTransform(int viewWidth, int viewHeight) {
        if (null == glesView || null == previewSize) {
            return;
        }
        int rotation = getWindowManager().getDefaultDisplay().getRotation();
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
        glesView.setTransform(matrix);
    }

    /** Compares two {@code Size}s based on their areas. */
    private static class CompareSizesByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum(
                    (long) lhs.getWidth() * lhs.getHeight() - (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    /**
     * Sets up member variables related to camera.
     *
     * @param width The width of available size for camera preview
     * @param height The height of available size for camera preview
     */
    private void setUpCameraOutputs(int width, int height) {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
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
                int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
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
                getWindowManager().getDefaultDisplay().getSize(displaySize);
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

                previewSize =
                        chooseOptimalSize(
                                map.getOutputSizes(SurfaceTexture.class),
                                rotatedPreviewWidth,
                                rotatedPreviewHeight,
                                maxPreviewWidth,
                                maxPreviewHeight,
                                largest);

                // We fit the aspect ratio of TextureView to the size of preview we picked.
                int orientation = getResources().getConfiguration().orientation;
                if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
                    glesView.setAspectRatio(previewSize.getWidth(), previewSize.getHeight());
                } else {
                    glesView.setAspectRatio(previewSize.getHeight(), previewSize.getWidth());
                }

                this.cameraId = cameraId;
                return;
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, "Failed to access Camera", e);
        } catch (NullPointerException e) {
            // Currently an NPE is thrown when the Camera2API is used but not supported on the
            // device this code runs.
            //ErrorDialog.newInstance(getString(R.string.camera_error))
                    //.show(getChildFragmentManager(), FRAGMENT_DIALOG);
        }
    }

    /**
     * Resizes image.
     *
     * Attempting to use too large a preview size could  exceed the camera bus' bandwidth limitation,
     * resulting in gorgeous previews but the storage of garbage capture data.
     *
     * Given {@code choices} of {@code Size}s supported by a camera, choose the smallest one that is
     * at least as large as the respective texture view size, and that is at most as large as the
     * respective max size, and whose aspect ratio matches with the specified value. If such size
     * doesn't exist, choose the largest one that is at most as large as the respective max size, and
     * whose aspect ratio matches with the specified value.
     *
     * @param choices The list of sizes that the camera supports for the intended output class
     * @param textureViewWidth The width of the texture view relative to sensor coordinate
     * @param textureViewHeight The height of the texture view relative to sensor coordinate
     * @param maxWidth The maximum width that can be chosen
     * @param maxHeight The maximum height that can be chosen
     * @param aspectRatio The aspect ratio
     * @return The optimal {@code Size}, or an arbitrary one if none were big enough
     */
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

    @Override
    protected void onPause() {
        super.onPause();
        stopBackgroundThread();
    }

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        // Start the classification train & load an initial model.
        synchronized (lock) {
            // only run when there is a frame available
            runSegment = false;
        }
        backgroundHandler.post(periodicSegment);
        updateActiveModel();
    }

    private void updateActiveModel() {
        backgroundHandler.post(new Runnable() {
            @Override
            public void run() {
                try {
                    recreateSegmenter(getModel(), getDevice(), getNumThreads());
                } catch (Exception e) {
                    segment = null;
                }

                if (segment == null) {
                    return;
                }

                segment.setNumThreads(getNumThreads());
                EGLDisplay[] display = new EGLDisplay[1];
                EGLContext[] context = new EGLContext[1];
                EGLConfig[] config  = new EGLConfig [1];
                EGLSurface[] surface = new EGLSurface[1];

                initGLES(context, display, config, surface, eglContext);
                gpuDisplay = display[0];
                gpuContext = context[0];
                gpuConfig  = config [0];
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

    private Runnable periodicSegment =
            new Runnable() {
                @Override
                public void run() {
                    synchronized (lock) {
                        if (runSegment) {
                            if (segment != null) {
                                if (segment.getInputSsboId() == 0) {
                                    //
                                } else {
                                    // wait until next frame in the SurfaceTexture frame listener in onFrameAvailable()
                                    segmentFrameSSBO();
                                    runSegment = false;
                                }
                            }
                        }
                    }
                    backgroundHandler.post(periodicSegment);
                }
            };

    private void segmentFrameSSBO() {
        if (segment == null || cameraDevice == null) {
            // It's important to not call showToast every frame, or else the app will starve and
            // hang. updateActiveModel() already puts a error message up with showToast.
            // showToast("Uninitialized Classifier or invalid context.");
            return;
        }
        camSurfTex.updateTexImage();

        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);
        // copy the surface to texture
        long copy_t0 = SystemClock.uptimeMillis();
        copyCamTexToSsbo();
        long copy_t1 = SystemClock.uptimeMillis();

        // classify the frame in the SSBO
        EGL14.eglMakeCurrent(gpuDisplay, gpuSurface, gpuSurface, gpuContext);
        classifier.classifyFrameSSBO(textToShow, copy_t1 - copy_t0);

        // resumes the normal egl context
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);

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
        SurfaceTexture outputSurf = glesView.getSurfaceTexture();
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

    // create a texture name
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
