package eto.cli;

import eto.nn.ModelConfig;
import eto.nn.TensorParam;
import eto.nn.TinyLlama;

import java.util.Random;

/** Finite-difference check of the analytic backprop. Run: java -cp build eto.cli.GradCheck */
public final class GradCheck {
    private GradCheck() {
    }

    public static void main(String[] args) {
        ModelConfig cfg = new ModelConfig(9, 8, 1, 2, 16, 8, 1e-5f, 10000f);
        TinyLlama model = new TinyLlama(cfg);
        model.initRandom(new Random(7), 0.05f);
        int[] tokens = {1, 2, 3, 4, 5, 6};
        model.zeroGrad();
        float base = model.lossAndBackward(tokens);
        float eps = args.length > 0 ? Float.parseFloat(args[0]) : 2e-3f;
        double maxRel = 0.0;
        double maxRelBig = 0.0;
        double maxAbs = 0.0;
        String worst = "";
        String worstAbs = "";
        long checked = 0;
        java.util.List<String> topAbs = new java.util.ArrayList<>();
        java.util.List<String> topRel = new java.util.ArrayList<>();
        for (TensorParam p : model.params()) {
            for (int i = 0; i < p.length(); i++) {
                float analytic = p.g()[i];
                float save = p.w()[i];
                p.w()[i] = save + eps;
                float lp = model.loss(tokens);
                p.w()[i] = save - eps;
                float lm = model.loss(tokens);
                p.w()[i] = save;
                float numeric = (lp - lm) / (2f * eps);
                float abs = Math.abs(numeric - analytic);
                float denom = Math.max(1e-4f, Math.max(Math.abs(numeric), Math.abs(analytic)));
                float rel = abs / denom;
                String row = p.name() + "[" + i + "] analytic=" + analytic + " numeric=" + numeric;
                if (abs > maxAbs) {
                    maxAbs = abs;
                    worstAbs = row;
                }
                if (rel > maxRel) {
                    maxRel = rel;
                    worst = row;
                }
                if (Math.max(Math.abs(numeric), Math.abs(analytic)) > 0.01f && rel > maxRelBig) {
                    maxRelBig = rel;
                }
                if (abs > 1e-3f) {
                    topAbs.add(String.format("abs=%.5f rel=%.4f %s", abs, rel, row));
                }
                if (rel > 0.05f && Math.max(Math.abs(numeric), Math.abs(analytic)) > 1e-3f) {
                    topRel.add(String.format("rel=%.4f abs=%.5f %s", rel, abs, row));
                }
                checked++;
            }
        }
        System.out.printf("loss=%.6f  checked=%d  maxRelErr=%.6f  maxRelErr(|g|>0.01)=%.6f  maxAbsErr=%.6f%n",
                base, checked, maxRel, maxRelBig, maxAbs);
        System.out.println("worst rel : " + worst);
        System.out.println("worst abs : " + worstAbs);
        topAbs.sort(java.util.Comparator.reverseOrder());
        topRel.sort(java.util.Comparator.reverseOrder());
        System.out.println("-- top absolute errors --");
        topAbs.stream().limit(10).forEach(System.out::println);
        System.out.println("-- top relative errors (|g|>1e-3) --");
        topRel.stream().limit(10).forEach(System.out::println);
        if (maxAbs > 5e-4 || maxRelBig > 0.02) {
            System.out.println("GRADCHECK: FAIL");
            System.exit(1);
        }
        System.out.println("GRADCHECK: OK");
    }
}
