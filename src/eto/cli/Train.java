package eto.cli;

import eto.train.Trainer;

import java.nio.file.Path;

/** Trainer entry point. */
public final class Train {
    private Train() {
    }

    public static void main(String[] args) throws Exception {
        Args a = Args.parse(args);
        if (a.flag("help")) {
            usage();
            return;
        }
        Path resume = a.has("resume") ? Path.of(a.get("resume")) : null;
        Path checkpoint = a.has("checkpoint") ? Path.of(a.get("checkpoint")) : resume;
        Trainer.Options o = new Trainer.Options(
                Path.of(a.get("data", "data/sample.txt")),
                Path.of(a.get("out", "models/eto-tiny.gguf")),
                a.get("name", "eto-tiny"),
                a.getInt("steps", 400),
                a.getInt("batch", 4),
                a.getInt("ctx", 64),
                a.getInt("dim", 64),
                a.getInt("layers", 2),
                a.getInt("heads", 4),
                a.getInt("ffn", 128),
                a.getFloat("lr", 3e-3f),
                a.getLong("seed", 1337),
                a.getFloat("std", 0.02f),
                checkpoint,
                resume,
                a.getInt("checkpoint-every", 100));
        Trainer.train(o);
    }

    private static void usage() {
        System.out.println("""
                Usage: ./trainer.sh [options]

                  --data PATH      text file used as raw-byte training data (default data/sample.txt)
                  --out PATH       output GGUF (default models/eto-tiny.gguf)
                  --name NAME      model name written into metadata (default eto-tiny)
                  --steps N        optimizer steps (default 400)
                  --batch N        sequences per step (default 4)
                  --ctx N          context length (default 64)
                  --dim N          embedding width (default 64)
                  --layers N       transformer blocks (default 2)
                  --heads N        attention heads (default 4)
                  --ffn N          SwiGLU hidden width (default 128)
                  --lr F           peak learning rate (default 0.003)
                  --seed N         RNG seed (default 1337)
                  --checkpoint PATH  save resumable training state every N steps
                  --resume PATH      restore model and AdamW state before training
                  --checkpoint-every N  checkpoint interval (default 100)
                """);
    }
}
