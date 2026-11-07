import Lake
open Lake DSL

package LLMComputer where
  version := v!"0.1.0"
  keywords := #["llm", "transformer", "wasm", "compiler"]

require TinyGrad4 from "../tinygrad"

@[default_target]
lean_lib LLMComputer where
  globs := #[`LLMComputer, `LLMComputer.Types, `LLMComputer.Vocab,
             `LLMComputer.Attention, `LLMComputer.Compiler,
             `LLMComputer.Execute, `LLMComputer.DSL]
  precompileModules := true

lean_exe llmc_demo where
  root := `LLMComputer.Examples
  moreLinkArgs :=
    if System.Platform.isOSX then
      #["-Wl,-syslibroot,/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
        "-framework", "Metal", "-framework", "Foundation", "-framework", "Accelerate",
        "-lobjc"]
    else #[]
