package com.coocoo.tflite.activity;

import android.app.Activity;
import android.os.Bundle;

import androidx.annotation.Nullable;

import com.coocoo.tflite.R;
import com.coocoo.tflite.fragment.Camera2BasicFragment;
import com.coocoo.tflite.fragment.SegmentFragment;

public class SegmentGLActivity extends Activity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_segmentgl);
        if (null == savedInstanceState) {
            getFragmentManager()
                    .beginTransaction()
                    .replace(R.id.container, Camera2BasicFragment.newInstance())
                    .commit();
        }
    }
}
