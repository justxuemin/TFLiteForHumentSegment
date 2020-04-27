package com.coocoo.tflite.activity;

import android.app.Activity;
import android.os.Bundle;
import android.util.DisplayMetrics;

import androidx.annotation.Nullable;

import com.coocoo.tflite.MyCamera;
import com.coocoo.tflite.MyGLSurfaceView;

public class GLActivity extends Activity {

    private MyCamera mCamera;
    private MyGLSurfaceView mGLSurfaceView;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mGLSurfaceView = new MyGLSurfaceView(this);
        DisplayMetrics dm = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(dm);
        mCamera = new MyCamera(this);
        mCamera.setupCamera(dm.widthPixels, dm.heightPixels);
        if (!mCamera.openCamera()) {
            return;
        }
        mGLSurfaceView.init(mCamera, false, this);
        setContentView(mGLSurfaceView);
    }
}
