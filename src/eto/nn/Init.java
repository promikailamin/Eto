package eto.nn;

import java.util.Random;

public final class Init {
    private Init() {
    }

    public static void normal(Random rng, float[] data, float std) {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (rng.nextGaussian() * std);
        }
    }

    public static void fill(float[] data, float x) {
        java.util.Arrays.fill(data, x);
    }
}
