package com.coocoo.tflite;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;

public class MyGLSurfaceView extends GLSurfaceView {


    private MyRender mRender;

    public MyGLSurfaceView(Context context) {
        super(context);
    }

    public MyGLSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }


    public void init(MyCamera camera, boolean isPreviewStarted, Context context) {
        setEGLContextClientVersion(2);

        mRender = new MyRender();
        mRender.init(this, camera, isPreviewStarted, context);
        setRenderer(mRender);
    }
}
