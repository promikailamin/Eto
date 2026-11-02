eto includes pure-Java implementations derived from the public GGUF/GGML
format specifications and reference dequantization routines.

llama.cpp / ggml
----------------
Copyright (c) 2023-2026 The ggml authors
License: MIT
https://github.com/ggml-org/llama.cpp

The C reference code was consulted for the GGUF container layout, ggml tensor
type sizes, and the dequantization formulas in
src/eto/gguf/Quant.java. The Java code in this repository is an independent
implementation; no C source is compiled, linked, or invoked at runtime.

Everything else in this repository is original code released under the same
MIT terms unless noted otherwise.
