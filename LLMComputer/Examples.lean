import Float64
import TinyGrad4
import LLMComputer.Types
import LLMComputer.Vocab
import LLMComputer.Attention
import LLMComputer.Compiler
import LLMComputer.Execute
import LLMComputer.DSL

/-!
# LLMComputer.Examples

Demonstration programs for the LLM Computer.

Each example is presented in three ways:
1. **DSL** — the `llm_wasm { ... }` notation (human-readable)
2. **Native interpreter** — ground-truth execution in pure Lean 4
3. **LLM Computer** — execution via the analytically compiled transformer

## Examples

| Program          | Description                               |
|------------------|-------------------------------------------|
| `addExample`     | Compute 5 + 3 = 8                         |
| `mulExample`     | Compute 6 × 7 = 42                        |
| `quadraticEx`    | Evaluate 2×² + 3×  + 1 at × = 4          |
| `gcdExample`     | Compute GCD(48, 36) = 12 (via locals)     |
| `sumExample`     | Compute Σ 1..10 = 55 (unrolled)           |
| `fibExample`     | Compute fib(7) = 13 (unrolled iteration)  |
| `collatzExample` | Collatz steps for n=6 → 8 steps           |
-/

namespace LLMComputer.Examples

open LLMComputer LLMComputer.DSL
open LLMComputer.Execute
open LLMComputer.Compiler
open TinyGrad4

/-! ## Example 1: Simple addition (5 + 3 = 8) -/

/-- 5 + 3 = 8 -/
def addFunc : WasmFunc := llm_wasm {
  i32.const 5
  i32.const 3
  i32.add
  ret
}

def addModule : WasmModule := llm_wasm_module addFunc

/-! ## Example 2: Multiplication (6 × 7 = 42) -/

def mulFunc : WasmFunc := llm_wasm {
  i32.const 6
  i32.const 7
  i32.mul
  ret
}

def mulModule : WasmModule := llm_wasm_module mulFunc

/-! ## Example 3: Quadratic expression 2×² + 3× + 1 at × = 4

    Push 4, compute x*x = 16, *2 = 32, then x*3 = 12, then +1.
    Result: 32 + 12 + 1 = 45.
-/

def quadFunc : WasmFunc := llm_wasm {
  i32.const 4     -- x
  dup             -- x, x
  i32.mul         -- x²
  i32.const 2
  i32.mul         -- 2x²
  i32.const 4     -- push x again (no local vars in this simple version)
  i32.const 3
  i32.mul         -- 3x
  i32.add         -- 2x² + 3x
  i32.const 1
  i32.add         -- 2x² + 3x + 1
  ret
}

def quadModule : WasmModule := llm_wasm_module quadFunc

/-! ## Example 4: GCD via Euclidean algorithm (stored in locals)

    Compute gcd(48, 36) = 12.
    Uses local variables 0 (a) and 1 (b).
    Euclidean: while b ≠ 0 { (a, b) = (b, a mod b) }
    Flat implementation (loop unrolled to fixed depth for simplicity):
-/

/-- GCD(48, 36) = 12 via two iterations of the Euclidean algorithm (no loop; hardcoded). -/
def gcdFunc : WasmFunc := llm_wasm {
  -- Store a=48 in local 0, b=36 in local 1
  i32.const 48    local.set 0
  i32.const 36    local.set 1
  -- Iteration 1: (a, b) = (36, 48 mod 36) = (36, 12)
  local.get 0    local.get 1    i32.rem_s    -- 48 mod 36 = 12
  local.get 1                                -- [12, 36]
  local.set 0    local.set 1                 -- a=36, b=12
  -- Iteration 2: (a, b) = (12, 36 mod 12) = (12, 0)
  local.get 0    local.get 1    i32.rem_s    -- 36 mod 12 = 0
  local.get 1                                -- [0, 12]
  local.set 0    local.set 1                 -- a=12, b=0 (done)
  -- b=0 → result is a=12
  local.get 0    ret
}

def gcdModule : WasmModule := llm_wasm_module { gcdFunc with nLocals := 2, name := "gcd" }

/-! ## Example 5: Sum 1..n for n=10 (= 55) using locals -/

def sumFunc : WasmFunc := llm_wasm {
  -- acc = 0, i = 1
  i32.const 0
  local.set 0    -- acc = 0
  i32.const 1
  local.set 1    -- i = 1
  -- Unrolled loop: sum 1..10
  local.get 0    local.get 1    i32.add    local.set 0    -- acc += 1
  i32.const 2    local.get 0    i32.add    local.set 0    -- acc += 2
  i32.const 3    local.get 0    i32.add    local.set 0    -- acc += 3
  i32.const 4    local.get 0    i32.add    local.set 0    -- acc += 4
  i32.const 5    local.get 0    i32.add    local.set 0    -- acc += 5
  i32.const 6    local.get 0    i32.add    local.set 0    -- acc += 6
  i32.const 7    local.get 0    i32.add    local.set 0    -- acc += 7
  i32.const 8    local.get 0    i32.add    local.set 0    -- acc += 8
  i32.const 9    local.get 0    i32.add    local.set 0    -- acc += 9
  i32.const 10   local.get 0    i32.add    local.set 0    -- acc += 10
  local.get 0    -- push result
  ret
}

def sumModule : WasmModule := llm_wasm_module { sumFunc with nLocals := 2, name := "sum_1_to_10" }

/-! ## Example 6: Fibonacci(7) = 13 via unrolled iteration

    Iterative Fibonacci: (f0, f1) = (0, 1), repeat 6 times: (f0, f1) ← (f1, f0+f1).
    Result: f1 = fib(7) = 13.

    Percepta's blog demo program: "fibonacci"
-/

/-- fib(7) = 13 via 6 unrolled iterations of the Fibonacci recurrence. -/
def fibFunc : WasmFunc := llm_wasm {
  -- locals: 0 = f0, 1 = f1
  i32.const 0  local.set 0   -- f0 = 0
  i32.const 1  local.set 1   -- f1 = 1
  -- iter 1: t=f0+f1=1, f0=f1=1, f1=t=1  → (f0,f1) = (1,1)
  local.get 0  local.get 1  i32.add  local.get 1  local.set 0  local.set 1
  -- iter 2: t=1+1=2, f0=1, f1=2        → (f0,f1) = (1,2)
  local.get 0  local.get 1  i32.add  local.get 1  local.set 0  local.set 1
  -- iter 3: t=1+2=3, f0=2, f1=3        → (f0,f1) = (2,3)
  local.get 0  local.get 1  i32.add  local.get 1  local.set 0  local.set 1
  -- iter 4: t=2+3=5, f0=3, f1=5        → (f0,f1) = (3,5)
  local.get 0  local.get 1  i32.add  local.get 1  local.set 0  local.set 1
  -- iter 5: t=3+5=8, f0=5, f1=8        → (f0,f1) = (5,8)
  local.get 0  local.get 1  i32.add  local.get 1  local.set 0  local.set 1
  -- iter 6: t=5+8=13, f0=8, f1=13      → (f0,f1) = (8,13)
  local.get 0  local.get 1  i32.add  local.get 1  local.set 0  local.set 1
  local.get 1  ret  -- return fib(7) = 13
}

def fibModule : WasmModule := llm_wasm_module { fibFunc with nLocals := 2, name := "fib7" }

/-! ## Example 7: Collatz steps for n=6 = 8

    Collatz sequence: if n even → n/2; if n odd → 3n+1. Count steps to reach 1.
    n=6: 6→3→10→5→16→8→4→2→1  (8 steps)

    Unrolled: we know the parity of n at each step, so we apply
    the correct rule explicitly without branches.

    Percepta's blog demo program: "collatz"
-/

/-- Count the 8 steps of Collatz(6): 6→3→10→5→16→8→4→2→1. -/
def collatzFunc : WasmFunc := llm_wasm {
  -- locals: 0=n, 1=steps
  i32.const 6  local.set 0   -- n = 6
  i32.const 0  local.set 1   -- steps = 0
  -- step 1: n=6 (even) → n = 6/2 = 3, steps = 1
  local.get 0  i32.const 2  i32.div_s  local.set 0
  local.get 1  i32.const 1  i32.add    local.set 1
  -- step 2: n=3 (odd) → n = 3*3+1 = 10, steps = 2
  i32.const 3  local.get 0  i32.mul  i32.const 1  i32.add  local.set 0
  local.get 1  i32.const 1  i32.add  local.set 1
  -- step 3: n=10 (even) → n = 10/2 = 5, steps = 3
  local.get 0  i32.const 2  i32.div_s  local.set 0
  local.get 1  i32.const 1  i32.add    local.set 1
  -- step 4: n=5 (odd) → n = 3*5+1 = 16, steps = 4
  i32.const 3  local.get 0  i32.mul  i32.const 1  i32.add  local.set 0
  local.get 1  i32.const 1  i32.add  local.set 1
  -- step 5: n=16 (even) → n = 16/2 = 8, steps = 5
  local.get 0  i32.const 2  i32.div_s  local.set 0
  local.get 1  i32.const 1  i32.add    local.set 1
  -- step 6: n=8 (even) → n = 8/2 = 4, steps = 6
  local.get 0  i32.const 2  i32.div_s  local.set 0
  local.get 1  i32.const 1  i32.add    local.set 1
  -- step 7: n=4 (even) → n = 4/2 = 2, steps = 7
  local.get 0  i32.const 2  i32.div_s  local.set 0
  local.get 1  i32.const 1  i32.add    local.set 1
  -- step 8: n=2 (even) → n = 2/2 = 1, steps = 8
  local.get 0  i32.const 2  i32.div_s  local.set 0
  local.get 1  i32.const 1  i32.add    local.set 1
  -- result: 8 steps
  local.get 1  ret
}

def collatzModule : WasmModule := llm_wasm_module { collatzFunc with nLocals := 2, name := "collatz6" }

/-! ## Running and displaying examples -/

/-- Display the token encoding of a module -/
def describeTokens (m : WasmModule) (label : String) : IO Unit := do
  let toks := Vocab.encodeModule m
  IO.println s!"{label}: {toks.size} tokens = {showTokens m}"

/-- Run native interpreter and print result -/
def runAndShow (label : String) (m : WasmModule) (expected : Int) : IO Bool := do
  match runNative m with
  | .ok v =>
    let pass := v == expected
    let check := if pass then "✓" else "✗"
    IO.println s!"[{check}] {label}: native = {v} (expected {expected})"
    pure pass
  | .error e =>
    IO.println s!"[✗] {label}: ERROR: {e}"
    pure false

/-- Compile and display basic info about the LLM Computer weights -/
def showCompiled (label : String) (m : WasmModule) : IO Unit := do
  let weights := runUOpM (Compiler.compile m)
  IO.println s!"  {label}: compiled {weights.layers.size} layers, vocab {LLMDims.default.vocabSize}"

/-! ## Main demo entry point -/

def main : IO Unit := do
  IO.println "═══════════════════════════════════════════"
  IO.println "  LLM Computer — Analytical Code-to-Weights"
  IO.println "  Based on Percepta AI (March 2026)"
  IO.println "═══════════════════════════════════════════"
  IO.println ""
  IO.println "Architecture:"
  IO.println s!"  layers   = {LLMDims.default.layers}"
  IO.println s!"  d_model  = {LLMDims.default.dModel}"
  IO.println s!"  heads    = {LLMDims.default.heads} × {LLMDims.default.headDim} dims"
  IO.println s!"  vocab    = {LLMDims.default.vocabSize} tokens"
  IO.println s!"  KV cache = HullKVCache (O(log n) for arithmetic traces)"
  IO.println ""
  IO.println "── Native interpreter (ground truth) ──────"
  let mut allPass := true
  allPass := allPass && (← runAndShow "5 + 3"          addModule     8)
  allPass := allPass && (← runAndShow "6 × 7"          mulModule     42)
  allPass := allPass && (← runAndShow "2×4² + 3×4 + 1" quadModule    45)
  allPass := allPass && (← runAndShow "GCD(48,36)"     gcdModule     12)
  allPass := allPass && (← runAndShow "Σ 1..10"        sumModule     55)
  allPass := allPass && (← runAndShow "fib(7)"         fibModule     13)
  allPass := allPass && (← runAndShow "collatz(6)"     collatzModule  8)
  IO.println ""
  IO.println "── Token encoding ──────────────────────────"
  describeTokens addModule     "add"
  describeTokens mulModule     "mul"
  describeTokens quadModule    "quad"
  describeTokens fibModule     "fib7"
  describeTokens collatzModule "collatz6"
  IO.println ""
  IO.println "── Analytical compilation ──────────────────"
  showCompiled "add"     addModule
  showCompiled "mul"     mulModule
  showCompiled "quad"    quadModule
  showCompiled "fib7"    fibModule
  showCompiled "collatz" collatzModule
  IO.println ""
  if allPass then
    IO.println "All native tests passed ✓"
  else
    IO.println "Some tests failed ✗"
    IO.Process.exit 1

end LLMComputer.Examples

def main : IO Unit := LLMComputer.Examples.main
