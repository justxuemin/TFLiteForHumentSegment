package com.coocoo.tflite.tf;

import android.app.Activity;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

public class SegmentFromShiShuai extends Segment {


    private static final float IMAGE_MEAN = -1.0f;
    private static final float IMAGE_STD = 1.0f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = -1.0f;

    private static final float PROBABILITY_STD = 1.0f;

    public SegmentFromShiShuai(Activity activity, Device device, int numThreads) throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() {
        return "shishuai.tflite";
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
}
