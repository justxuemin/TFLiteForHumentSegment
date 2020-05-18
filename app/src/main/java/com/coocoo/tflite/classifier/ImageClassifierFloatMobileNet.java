/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.coocoo.tflite.classifier;

import android.app.Activity;
import java.io.IOException;

/** This classifier works with the float MobileNet model. */
public class ImageClassifierFloatMobileNet extends ImageClassifier {

  /** The mobile net requires additional normalization of the used input. */
  private static final float IMAGE_MEAN = 127.5f;

  private static final float IMAGE_STD = 127.5f;

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][] labelProbArray = null;

  /**
   * Initializes an {@code ImageClassifierFloatMobileNet}.
   *
   * @param activity
   */
  public ImageClassifierFloatMobileNet(Activity activity) throws IOException {
    super(activity);
    labelProbArray = new float[1][getNumLabels()];
  }

  @Override
  public String getModelPath() {
    // you can download this file from
    // see build.gradle for where to obtain this file. It should be auto
    // downloaded into assets.
    return "mobilenet_v1_1.0_224.tflite";
  }

  @Override
  public String getLabelPath() {
    return "labels_mobilenet_quant_v1_224.txt";
  }

  @Override
  public int getImageSizeX() {
    return 224;
  }

  @Override
  public int getImageSizeY() {
    return 224;
  }

  @Override
  public int getNumBytesPerChannel() {
    return 4; // Float.SIZE / Byte.SIZE;
  }

  @Override
  public void addPixelValue(int pixelValue) {
    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
  }

  @Override
  public float getProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  public void setProbability(int labelIndex, Number value) {
    labelProbArray[0][labelIndex] = value.floatValue();
  }

  @Override
  public float getNormalizedProbability    (int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  public void runInference() {
    if (getInputSsboId() == 0) {
      tflite.run(imgData, labelProbArray);
    } else {
      // run inference for the pixels in SSBO
      tflite.run(null, labelProbArray);
    }
  }
}
