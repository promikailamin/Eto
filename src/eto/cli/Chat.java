package eto.cli;

import eto.infer.Generator;
import eto.infer.LlamaModel;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Interactive CLI chatbot with persistent conversation history. */
public final class Chat {
    private Chat() {
    }

    public static void main(String[] args) throws Exception {
        Args a = Args.parse(args);
        if (a.flag("help")) {
            usage();
            return;
        }
        Path modelPath = Path.of(a.get("model", "models/eto-tiny.gguf"));
        int ctxArg = a.getInt("ctx", 0);
        int maxNew = a.getInt("tokens", 256);
        float temp = a.flag("greedy") ? 0f : a.getFloat("temp", 0.8f);
        int topK = a.getInt("top-k", 40);
        float topP = a.getFloat("top-p", 0.95f);
        float repeat = a.getFloat("repeat", -1f);
        long seed = a.getLong("seed", 42);
        String system = a.get("system", "");
        String oneShot = a.has("prompt") ? a.get("prompt") : null;

        try (LlamaModel model = LlamaModel.open(modelPath)) {
            int ctx = ctxArg > 0 ? Math.min(ctxArg, model.cfg.ctxLen) : Math.min(model.cfg.ctxLen, 1024);
            if (repeat < 0f) {
                // A byte-level model (vocab ~258) must not use a repetition penalty: every letter
                // is in the recent window, so the penalty just distorts the character stream.
                repeat = model.tokenizer.vocabSize() <= 300 ? 1.0f : 1.1f;
            }
            String templateName = a.get("template", "auto");
            if ("auto".equalsIgnoreCase(templateName)
                    && "Eto".equalsIgnoreCase(model.reader.getString("general.name", ""))) {
                templateName = "eto";
            }
            ChatTemplate.Kind template = a.flag("raw")
                    ? ChatTemplate.Kind.RAW
                    : ChatTemplate.parse(templateName, model.tokenizer,
                            model.reader.getString("tokenizer.chat_template", null));

            System.out.println("model    : " + modelPath);
            System.out.println("config   : " + model.cfg);
            System.out.println("tokenizer: " + model.tokenizer.family() + " (vocab " + model.tokenizer.vocabSize() + ")");
            System.out.println("template : " + template + (a.has("template") ? "" : " (auto)"));
            System.out.println("context  : " + ctx + " tokens, max new " + maxNew);
            System.out.println("type /help for commands, /exit to quit");
            System.out.println();

            List<ChatTemplate.Turn> turns = new ArrayList<>();
            int turnSeed = 0;

            if (oneShot != null) {
                runTurn(model, template, system, turns, oneShot, ctx, maxNew,
                        new Generator.Params(temp, topK, topP, repeat, seed + turnSeed));
                return;
            }

            BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            while (true) {
                System.out.print(">>> ");
                System.out.flush();
                String line = in.readLine();
                if (line == null) {
                    System.out.println();
                    break;
                }
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                if (line.startsWith("/")) {
                    String[] parts = line.split("\\s+", 2);
                    String cmd = parts[0].toLowerCase();
                    String rest = parts.length > 1 ? parts[1] : "";
                    switch (cmd) {
                        case "/exit", "/quit", "/bye" -> {
                            System.out.println("bye");
                            return;
                        }
                        case "/reset" -> {
                            turns.clear();
                            System.out.println("[history cleared]");
                        }
                        case "/system" -> {
                            system = rest;
                            System.out.println("[system prompt set]");
                        }
                        case "/temp" -> {
                            temp = Float.parseFloat(rest);
                            System.out.println("[temperature = " + temp + "]");
                        }
                        case "/topk" -> {
                            topK = Integer.parseInt(rest);
                            System.out.println("[top-k = " + topK + "]");
                        }
                        case "/topp" -> {
                            topP = Float.parseFloat(rest);
                            System.out.println("[top-p = " + topP + "]");
                        }
                        case "/repeat" -> {
                            repeat = Float.parseFloat(rest);
                            System.out.println("[repeat penalty = " + repeat + "]");
                        }
                        case "/tokens" -> {
                            maxNew = Integer.parseInt(rest);
                            System.out.println("[max new tokens = " + maxNew + "]");
                        }
                        case "/ctx" -> {
                            ctx = Math.min(Integer.parseInt(rest), model.cfg.ctxLen);
                            System.out.println("[context = " + ctx + "]");
                        }
                        case "/model" -> {
                            System.out.println(modelPath + "  " + model.cfg);
                            System.out.println("template=" + template + " ctx=" + ctx
                                    + " temp=" + temp + " top-k=" + topK + " top-p=" + topP
                                    + " repeat=" + repeat);
                        }
                        case "/help" -> help();
                        default -> System.out.println("[unknown command: " + cmd + "]");
                    }
                    continue;
                }
                try {
                    turns = runTurn(model, template, system, turns, line, ctx, maxNew,
                            new Generator.Params(temp, topK, topP, repeat, seed + turnSeed));
                    turnSeed++;
                    System.out.println();
                } catch (RuntimeException e) {
                    System.out.println("[generation error: " + e.getMessage() + "]");
                }
            }
        }
    }

    private static List<ChatTemplate.Turn> runTurn(
            LlamaModel model,
            ChatTemplate.Kind template,
            String system,
            List<ChatTemplate.Turn> turns,
            String user,
            int ctx,
            int maxNew,
            Generator.Params params) {
        List<ChatTemplate.Turn> working = new ArrayList<>(turns);
        int reserve = Math.min(maxNew, Math.max(8, ctx / 4));
        int[] ids;
        while (true) {
            String prompt = ChatTemplate.build(template, system, working, user);
            ids = model.tokenizer.encode(prompt, model.tokenizer.defaultAddBos());
            if (ids.length <= ctx - reserve || working.isEmpty()) {
                break;
            }
            working.remove(0);
        }
        if (ids.length >= ctx) {
            int keep = Math.max(1, ctx - 2);
            int[] trimmed = new int[keep];
            System.arraycopy(ids, ids.length - keep, trimmed, 0, keep);
            if (model.tokenizer.defaultAddBos()) {
                trimmed[0] = model.tokenizer.bosId();
            }
            ids = trimmed;
        }

        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        Generator.Result result = Generator.generate(model, ids, maxNew, ctx, params, (id, bytes) -> {
            try {
                System.out.write(bytes, 0, bytes.length);
                System.out.flush();
            } catch (Exception ignored) {
                // stdout closed
            }
            captured.write(bytes, 0, bytes.length);
        }, template == ChatTemplate.Kind.ETO ? new String[] {"\nuser:", "\n\n", "\nEto:"} : new String[0]);

        List<ChatTemplate.Turn> next = new ArrayList<>(working);
        next.add(new ChatTemplate.Turn(user, result.text()));
        return next;
    }

    private static void help() {
        System.out.println("""
                commands:
                  /exit, /quit      quit
                  /reset            clear conversation history
                  /system TEXT      set the system prompt
                  /temp F           sampling temperature (0 = greedy)
                  /topk N           top-k sampling
                  /topp F           nucleus sampling
                  /repeat F         repetition penalty
                  /tokens N         maximum new tokens per reply
                  /ctx N            context window (capped by the model)
                  /model            show model and sampling settings
                  /help             this help
                """);
    }

    private static void usage() {
        System.out.println("""
                Usage: ./runner.sh --model MODEL.gguf [options]

                  --model PATH     GGUF model (default models/eto-tiny.gguf)
                  --ctx N          context window (default min(model, 1024))
                  --tokens N       maximum new tokens per reply (default 256)
                  --temp F         temperature (default 0.8)
                  --top-k N        top-k (default 40)
                  --top-p F        nucleus (default 0.95)
                  --repeat F       repetition penalty (default 1.1; 1.0 for byte-level models)
                  --seed N         RNG seed (default 42)
                  --system TEXT    system prompt
                  --template NAME  auto|none|chatml|llama2|llama3|zephyr|alpaca|raw
                  --prompt TEXT    one-shot mode: answer once and exit
                  --greedy         temperature 0
                  --raw            alias for --template raw

                The runner loads your own eto GGUF as well as third-party
                llama / mistral / qwen2 GGUF files (F32/F16 and common quant types).
                """);
    }
}
