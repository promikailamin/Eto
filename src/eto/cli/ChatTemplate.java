package eto.cli;

import eto.tokenizer.Tokenizer;

import java.util.List;

/** Small, dependency-free chat prompt templates. */
public final class ChatTemplate {
    public enum Kind {
        ETO, NONE, CHATML, LLAMA2, LLAMA3, ZEPHYR, ALPACA, RAW
    }

    public record Turn(String user, String assistant) {
    }

    private ChatTemplate() {
    }

    public static Kind parse(String name, Tokenizer tok) {
        return parse(name, tok, null);
    }

    public static Kind parse(String name, Tokenizer tok, String chatTemplate) {
        if (name == null || name.isBlank() || "auto".equalsIgnoreCase(name)) {
            return auto(tok, chatTemplate);
        }
        return switch (name.toLowerCase()) {
            case "eto" -> Kind.ETO;
            case "none" -> Kind.NONE;
            case "chatml", "qwen" -> Kind.CHATML;
            case "llama2", "mistral" -> Kind.LLAMA2;
            case "llama3" -> Kind.LLAMA3;
            case "zephyr", "tinyllama" -> Kind.ZEPHYR;
            case "alpaca" -> Kind.ALPACA;
            case "raw" -> Kind.RAW;
            default -> throw new IllegalArgumentException("unknown template: " + name);
        };
    }

    public static Kind auto(Tokenizer tok) {
        return auto(tok, null);
    }

    public static Kind auto(Tokenizer tok, String chatTemplate) {
        String t = chatTemplate == null ? "" : chatTemplate;
        if (t.contains("<|im_start|>") || hasToken(tok, "<|im_start|>")) {
            return Kind.CHATML;
        }
        if (t.contains("<|start_header_id|>") || hasToken(tok, "<|start_header_id|>")
                || hasToken(tok, "<|eot_id|>")) {
            return Kind.LLAMA3;
        }
        if (t.contains("<|user|>") || hasToken(tok, "<|user|>")) {
            return Kind.ZEPHYR;
        }
        if (t.contains("[INST]") || hasToken(tok, "[INST]")) {
            return Kind.LLAMA2;
        }
        if (t.contains("### Instruction:") || hasToken(tok, "### Instruction:")) {
            return Kind.ALPACA;
        }
        return Kind.NONE;
    }

    private static boolean hasToken(Tokenizer tok, String text) {
        for (int i = 0; i < tok.vocabSize(); i++) {
            if (text.equals(tok.tokenText(i))) {
                return true;
            }
        }
        return false;
    }

    public static String build(Kind kind, String system, List<Turn> turns, String user) {
        return switch (kind) {
            case ETO -> eto(system, turns, user);
            case RAW -> user;
            case CHATML -> chatml(system, turns, user);
            case LLAMA2 -> llama2(system, turns, user);
            case LLAMA3 -> llama3(system, turns, user);
            case ZEPHYR -> zephyr(system, turns, user);
            case ALPACA -> alpaca(system, turns, user);
            case NONE -> plain(system, turns, user);
        };
    }

    private static String eto(String system, List<Turn> turns, String user) {
        StringBuilder sb = new StringBuilder();
        if (system != null && !system.isBlank()) {
            sb.append("system: ").append(system).append("\n\n");
        }
        for (Turn turn : turns) {
            sb.append("user: ").append(turn.user()).append("\n");
            sb.append("Eto: ").append(turn.assistant()).append("\n\n");
        }
        sb.append("user: ").append(user).append("\nEto:");
        return sb.toString();
    }

    private static String chatml(String system, List<Turn> turns, String user) {
        StringBuilder sb = new StringBuilder();
        if (system != null && !system.isBlank()) {
            sb.append("<|im_start|>system\n").append(system).append("<|im_end|>\n");
        }
        for (Turn t : turns) {
            sb.append("<|im_start|>user\n").append(t.user()).append("<|im_end|>\n");
            sb.append("<|im_start|>assistant\n").append(t.assistant()).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>user\n").append(user).append("<|im_end|>\n");
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }

    private static String llama3(String system, List<Turn> turns, String user) {
        StringBuilder sb = new StringBuilder();
        if (system != null && !system.isBlank()) {
            sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
                    .append(system).append("<|eot_id|>");
        }
        for (Turn t : turns) {
            sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
                    .append(t.user()).append("<|eot_id|>");
            sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                    .append(t.assistant()).append("<|eot_id|>");
        }
        sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
                .append(user).append("<|eot_id|>");
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return sb.toString();
    }

    private static String zephyr(String system, List<Turn> turns, String user) {
        StringBuilder sb = new StringBuilder();
        if (system != null && !system.isBlank()) {
            sb.append("<|system|>\n").append(system).append("</s>\n");
        }
        for (Turn t : turns) {
            sb.append("<|user|>\n").append(t.user()).append("</s>\n");
            sb.append("<|assistant|>\n").append(t.assistant()).append("</s>\n");
        }
        sb.append("<|user|>\n").append(user).append("</s>\n");
        sb.append("<|assistant|>\n");
        return sb.toString();
    }

    private static String llama2(String system, List<Turn> turns, String user) {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (Turn t : turns) {
            sb.append("[INST] ");
            if (first && system != null && !system.isBlank()) {
                sb.append("<<SYS>>\n").append(system).append("\n<</SYS>>\n\n");
            }
            sb.append(t.user()).append(" [/INST] ").append(t.assistant()).append(" </s>");
            first = false;
        }
        sb.append("[INST] ");
        if (first && system != null && !system.isBlank()) {
            sb.append("<<SYS>>\n").append(system).append("\n<</SYS>>\n\n");
        }
        sb.append(user).append(" [/INST]");
        return sb.toString();
    }

    private static String alpaca(String system, List<Turn> turns, String user) {
        StringBuilder sb = new StringBuilder();
        if (system != null && !system.isBlank()) {
            sb.append("### System:\n").append(system).append("\n\n");
        }
        for (Turn t : turns) {
            sb.append("### Instruction:\n").append(t.user()).append("\n\n");
            sb.append("### Response:\n").append(t.assistant()).append("\n\n");
        }
        sb.append("### Instruction:\n").append(user).append("\n\n### Response:\n");
        return sb.toString();
    }

    private static String plain(String system, List<Turn> turns, String user) {
        StringBuilder sb = new StringBuilder();
        if (system != null && !system.isBlank()) {
            sb.append(system).append("\n\n");
        }
        for (Turn t : turns) {
            sb.append("User: ").append(t.user()).append("\n");
            sb.append("Assistant: ").append(t.assistant()).append("\n\n");
        }
        sb.append("User: ").append(user).append("\nAssistant:");
        return sb.toString();
    }
}
