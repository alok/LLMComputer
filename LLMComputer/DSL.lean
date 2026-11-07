import Lean
import LLMComputer.Types
import LLMComputer.Vocab

/-!
# LLMComputer.DSL

Custom syntax macros for writing LLM Computer programs in a readable DSL.

## Quick start

```lean
import LLMComputer.DSL
open LLMComputer LLMComputer.DSL

-- Write a WebAssembly function body
def addFive : WasmFunc := llm_wasm {
  i32.const 5
  i32.const 3
  i32.add
  ret
}

-- Wrap in a module
def myModule : WasmModule := llm_module { program := addFive }

-- Run through native interpreter (ground truth)
#eval LLMComputer.Execute.runNative myModule  -- Except.ok 8
```

## DSL Grammar

```
llm_wasm { <instr>* }          — produces WasmFunc (body only)
llm_module { program := fn,    — produces WasmModule
             nLocals := n,
             name    := str }
```
-/

namespace LLMComputer.DSL

open Lean Parser

/-! ## Wasm instruction syntax category -/

declare_syntax_cat wasmInstr

-- Constants
syntax "i32.const" num      : wasmInstr
syntax "i32.const" "-" num  : wasmInstr   -- negative literal

-- Arithmetic
syntax "i32.add"   : wasmInstr
syntax "i32.sub"   : wasmInstr
syntax "i32.mul"   : wasmInstr
syntax "i32.div_s" : wasmInstr
syntax "i32.rem_s" : wasmInstr

-- Comparison (push 0 or 1)
syntax "i32.eq"    : wasmInstr
syntax "i32.ne"    : wasmInstr
syntax "i32.lt_s"  : wasmInstr
syntax "i32.gt_s"  : wasmInstr
syntax "i32.le_s"  : wasmInstr
syntax "i32.ge_s"  : wasmInstr

-- Bitwise
syntax "i32.and"   : wasmInstr
syntax "i32.or"    : wasmInstr
syntax "i32.xor"   : wasmInstr
syntax "i32.shl"   : wasmInstr
syntax "i32.shr_s" : wasmInstr

-- Stack ops
syntax "drop" : wasmInstr
syntax "dup"  : wasmInstr

-- Locals
syntax "local.get" num : wasmInstr
syntax "local.set" num : wasmInstr
syntax "local.tee" num : wasmInstr

-- Memory (with optional offset)
syntax "i32.load"          : wasmInstr   -- offset = 0
syntax "i32.load"  num     : wasmInstr
syntax "i32.store"         : wasmInstr   -- offset = 0
syntax "i32.store" num     : wasmInstr

-- Control flow
syntax "block"  num  : wasmInstr
syntax "loop"   num  : wasmInstr
syntax "end"         : wasmInstr
syntax "br"     num  : wasmInstr
syntax "br_if"  num  : wasmInstr
syntax "call"   num  : wasmInstr
syntax "return"      : wasmInstr
syntax "ret"         : wasmInstr  -- alias
syntax "nop"         : wasmInstr
syntax "unreachable" : wasmInstr

/-! ## `llm_wasm { ... }` macro → WasmFunc -/

syntax "llm_wasm" "{" wasmInstr* "}" : term

macro_rules
  | `(llm_wasm { $instrs:wasmInstr* }) => do
    let instrExprs ← instrs.mapM (fun instr => do
      match instr with
      | `(wasmInstr| i32.const $n:num)   => `(LLMComputer.WasmInstr.i32Const ($n : Int))
      | `(wasmInstr| i32.const - $n:num) => `(LLMComputer.WasmInstr.i32Const (- ($n : Int)))
      | `(wasmInstr| i32.add)   => `(LLMComputer.WasmInstr.i32Add)
      | `(wasmInstr| i32.sub)   => `(LLMComputer.WasmInstr.i32Sub)
      | `(wasmInstr| i32.mul)   => `(LLMComputer.WasmInstr.i32Mul)
      | `(wasmInstr| i32.div_s) => `(LLMComputer.WasmInstr.i32DivS)
      | `(wasmInstr| i32.rem_s) => `(LLMComputer.WasmInstr.i32RemS)
      | `(wasmInstr| i32.eq)    => `(LLMComputer.WasmInstr.i32Eq)
      | `(wasmInstr| i32.ne)    => `(LLMComputer.WasmInstr.i32Ne)
      | `(wasmInstr| i32.lt_s)  => `(LLMComputer.WasmInstr.i32LtS)
      | `(wasmInstr| i32.gt_s)  => `(LLMComputer.WasmInstr.i32GtS)
      | `(wasmInstr| i32.le_s)  => `(LLMComputer.WasmInstr.i32LeS)
      | `(wasmInstr| i32.ge_s)  => `(LLMComputer.WasmInstr.i32GeS)
      | `(wasmInstr| i32.and)   => `(LLMComputer.WasmInstr.i32And)
      | `(wasmInstr| i32.or)    => `(LLMComputer.WasmInstr.i32Or)
      | `(wasmInstr| i32.xor)   => `(LLMComputer.WasmInstr.i32Xor)
      | `(wasmInstr| i32.shl)   => `(LLMComputer.WasmInstr.i32Shl)
      | `(wasmInstr| i32.shr_s) => `(LLMComputer.WasmInstr.i32ShrS)
      | `(wasmInstr| drop)      => `(LLMComputer.WasmInstr.drop)
      | `(wasmInstr| dup)       => `(LLMComputer.WasmInstr.dup)
      | `(wasmInstr| local.get $n:num) => `(LLMComputer.WasmInstr.localGet ($n : Nat))
      | `(wasmInstr| local.set $n:num) => `(LLMComputer.WasmInstr.localSet ($n : Nat))
      | `(wasmInstr| local.tee $n:num) => `(LLMComputer.WasmInstr.localTee ($n : Nat))
      | `(wasmInstr| i32.load)         => `(LLMComputer.WasmInstr.i32Load 0)
      | `(wasmInstr| i32.load  $n:num) => `(LLMComputer.WasmInstr.i32Load ($n : Nat))
      | `(wasmInstr| i32.store)        => `(LLMComputer.WasmInstr.i32Store 0)
      | `(wasmInstr| i32.store $n:num) => `(LLMComputer.WasmInstr.i32Store ($n : Nat))
      | `(wasmInstr| block  $n:num)    => `(LLMComputer.WasmInstr.blockBegin ($n : Nat))
      | `(wasmInstr| loop   $n:num)    => `(LLMComputer.WasmInstr.loopBegin ($n : Nat))
      | `(wasmInstr| end)              => `(LLMComputer.WasmInstr.blockEnd)
      | `(wasmInstr| br     $n:num)    => `(LLMComputer.WasmInstr.br ($n : Nat))
      | `(wasmInstr| br_if  $n:num)    => `(LLMComputer.WasmInstr.brIf ($n : Nat))
      | `(wasmInstr| call   $n:num)    => `(LLMComputer.WasmInstr.call ($n : Nat))
      | `(wasmInstr| return)           => `(LLMComputer.WasmInstr.ret)
      | `(wasmInstr| ret)              => `(LLMComputer.WasmInstr.ret)
      | `(wasmInstr| nop)              => `(LLMComputer.WasmInstr.nop)
      | `(wasmInstr| unreachable)      => `(LLMComputer.WasmInstr.unreachable)
      | stx => Lean.Macro.throwError s!"llm_wasm: unrecognised instruction: {stx}")
    `({ body := #[$instrExprs,*] : LLMComputer.WasmFunc })

/-! ## Module builder (function, not macro — avoids keyword conflicts)

    Using a macro for `llm_module { ... }` would require registering `nLocals`,
    `name`, `memSize` as Lean 4 keywords, which breaks struct field syntax.
    Instead, provide simple constructor functions:

    ```lean
    llm_wasm_module fn               -- fn as sole function
    llm_wasm_module fn |>.setLocals 2  -- with 2 locals
    ```
-/

/-- Wrap a `WasmFunc` in a minimal `WasmModule` (1 page of memory). -/
def llm_wasm_module (fn : WasmFunc) : WasmModule :=
  { funcs := #[fn], memSize := 1, entryPoint := 0 }

/-! ## Token-stream display -/

/-- Show the token IDs that a WasmModule encodes to. -/
def showTokens (m : WasmModule) : String :=
  let toks := Vocab.encodeModule m
  "[" ++ (toks.foldl (fun acc t => if acc.isEmpty then s!"{t}" else acc ++ s!", {t}") "") ++ "]"

end LLMComputer.DSL
