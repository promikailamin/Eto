package eto.cli;

import java.util.Arrays;

/** Command dispatcher: train, chat, generate, info, gradcheck. */
public final class Main {
    private Main() {
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            usage();
            return;
        }
        String cmd = args[0];
        String[] rest = Arrays.copyOfRange(args, 1, args.length);
        try {
            switch (cmd) {
                case "train" -> Train.main(rest);
                case "chat", "run" -> Chat.main(rest);
                case "generate", "gen" -> Generate.main(rest);
                case "info" -> Info.main(rest);
                case "gradcheck" -> GradCheck.main(rest);
                case "help", "--help", "-h" -> usage();
                default -> {
                    System.err.println("unknown command: " + cmd);
                    usage();
                    System.exit(2);
                }
            }
        } catch (Exception e) {
            System.err.println("error: " + e.getMessage());
            if (System.getenv("ETO_DEBUG") != null) {
                e.printStackTrace();
            }
            System.exit(1);
        }
    }

    private static void usage() {
        System.out.println("""
                eto - a pure-Java tiny language model trainer and GGUF runner

                Commands:
                  train      train a byte-level llama-style model and write GGUF
                  chat       interactive chatbot (ollama-style REPL)
                  generate   one-shot text completion
                  info       inspect a GGUF file
                  gradcheck  finite-difference check of the trainer's backprop

                Quick start:
                  ./trainer.sh
                  ./runner.sh --model models/eto-tiny.gguf

                Run a command with --help for its options.
                """);
    }
}
