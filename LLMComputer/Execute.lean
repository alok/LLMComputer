import Float64
import TinyGrad4
import LLMComputer.Types
import LLMComputer.Vocab
import LLMComputer.Attention
import LLMComputer.Compiler

/-!
# LLMComputer.Execute

Token-by-token execution engine for the LLM Computer.

## Execution model

1. Embed the current token: hidden = E[token]
2. Run all transformer layers (each updates hidden + KV cache)
3. Compute logits: logits = hidden @ W_unembed
4. Argmax → next token
5. Terminate when HALT token is generated

## Two execution paths

- **Native** (`VMState.step`): reference Lean 4 interpreter for ground truth
- **Transformer** (`transformerStep`): the analytically compiled LLM Computer
-/

namespace LLMComputer.Execute

open TinyGrad4 StaticTensor Interpreter
open LLMComputer.Vocab
open LLMComputer.Attention
open LLMComputer.Compiler

/-! ## Stack helpers (Lean 4 has no Array.pop?) -/

/-- Pop the top element of an integer array stack.
    Returns `(top, remaining)` or `none` if empty. -/
private def stackPop (arr : Array Int) : Option (Int × Array Int) :=
  if arr.isEmpty then none
  else some (arr.back!, arr.pop)

/-- Array.get? substitute for arrays. -/
private def arrGet? (arr : Array Int) (idx : Nat) : Option Int :=
  if idx < arr.size then some arr[idx]! else none

/-! ## Native VM reference interpreter -/

/-- Execute one `WasmInstr` in the Lean 4 stack machine.
    Returns the updated VMState, or an error message. -/
def VMState.step (s : VMState) (instr : WasmInstr) : Except String VMState :=
  match instr with
  | .i32Const v =>
      .ok { s with stack := s.stack.push v }
  | .i32Add =>
      match stackPop s.stack with
      | none => .error "i32Add: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32Add: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (a + b) }
  | .i32Sub =>
      match stackPop s.stack with
      | none => .error "i32Sub: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32Sub: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (a - b) }
  | .i32Mul =>
      match stackPop s.stack with
      | none => .error "i32Mul: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32Mul: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (a * b) }
  | .i32DivS =>
      match stackPop s.stack with
      | none => .error "i32DivS: stack underflow"
      | some (b, rest) =>
        if b == 0 then .error "i32DivS: division by zero"
        else match stackPop rest with
          | none => .error "i32DivS: stack underflow"
          | some (a, rest2) => .ok { s with stack := rest2.push (a / b) }
  | .i32RemS =>
      match stackPop s.stack with
      | none => .error "i32RemS: stack underflow"
      | some (b, rest) =>
        if b == 0 then .error "i32RemS: division by zero"
        else match stackPop rest with
          | none => .error "i32RemS: stack underflow"
          | some (a, rest2) => .ok { s with stack := rest2.push (a % b) }
  | .i32Eq =>
      match stackPop s.stack with
      | none => .error "i32Eq: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32Eq: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (if a == b then 1 else 0) }
  | .i32Ne =>
      match stackPop s.stack with
      | none => .error "i32Ne: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32Ne: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (if a != b then 1 else 0) }
  | .i32LtS =>
      match stackPop s.stack with
      | none => .error "i32LtS: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32LtS: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (if a < b then 1 else 0) }
  | .i32GtS =>
      match stackPop s.stack with
      | none => .error "i32GtS: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32GtS: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (if a > b then 1 else 0) }
  | .i32LeS =>
      match stackPop s.stack with
      | none => .error "i32LeS: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32LeS: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (if a <= b then 1 else 0) }
  | .i32GeS =>
      match stackPop s.stack with
      | none => .error "i32GeS: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32GeS: stack underflow"
        | some (a, rest2) => .ok { s with stack := rest2.push (if a >= b then 1 else 0) }
  | .i32And =>
      match stackPop s.stack with
      | none => .error "i32And: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32And: stack underflow"
        | some (a, rest2) =>
          .ok { s with stack := rest2.push (Int.ofNat (a.toNat &&& b.toNat)) }
  | .i32Or =>
      match stackPop s.stack with
      | none => .error "i32Or: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32Or: stack underflow"
        | some (a, rest2) =>
          .ok { s with stack := rest2.push (Int.ofNat (a.toNat ||| b.toNat)) }
  | .i32Xor =>
      match stackPop s.stack with
      | none => .error "i32Xor: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32Xor: stack underflow"
        | some (a, rest2) =>
          .ok { s with stack := rest2.push (Int.ofNat (a.toNat ^^^ b.toNat)) }
  | .i32Shl =>
      match stackPop s.stack with
      | none => .error "i32Shl: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32Shl: stack underflow"
        | some (a, rest2) =>
          let shift := b.toNat % 32
          .ok { s with stack := rest2.push (a * (Int.ofNat (2 ^ shift))) }
  | .i32ShrS =>
      match stackPop s.stack with
      | none => .error "i32ShrS: stack underflow"
      | some (b, rest) =>
        match stackPop rest with
        | none => .error "i32ShrS: stack underflow"
        | some (a, rest2) =>
          let shift := b.toNat % 32
          .ok { s with stack := rest2.push (a / (Int.ofNat (2 ^ shift))) }
  | .drop =>
      match stackPop s.stack with
      | none => .error "drop: stack underflow"
      | some (_, rest) => .ok { s with stack := rest }
  | .dup =>
      match s.stack.back? with
      | none => .error "dup: empty stack"
      | some v => .ok { s with stack := s.stack.push v }
  | .localGet idx =>
      match arrGet? s.locals idx with
      | none => .error s!"localGet {idx}: out of range"
      | some v => .ok { s with stack := s.stack.push v }
  | .localSet idx =>
      match stackPop s.stack with
      | none => .error s!"localSet {idx}: stack underflow"
      | some (v, rest) =>
        let locs := if idx < s.locals.size then s.locals
                    else s.locals ++ Array.replicate (idx - s.locals.size + 1) 0
        .ok { s with stack := rest, locals := locs.set! idx v }
  | .localTee idx =>
      match s.stack.back? with
      | none => .error s!"localTee {idx}: empty stack"
      | some v =>
        let locs := if idx < s.locals.size then s.locals
                    else s.locals ++ Array.replicate (idx - s.locals.size + 1) 0
        .ok { s with locals := locs.set! idx v }
  | .i32Load offset =>
      match stackPop s.stack with
      | none => .error "i32Load: stack underflow"
      | some (addr, rest) =>
        let byteAddr := (addr + Int.ofNat offset).toNat
        if byteAddr + 3 >= s.memory.size then
          .error s!"i32Load: address {byteAddr} out of bounds"
        else
          let b0 := s.memory.getD byteAddr 0
          let b1 := s.memory.getD (byteAddr+1) 0
          let b2 := s.memory.getD (byteAddr+2) 0
          let b3 := s.memory.getD (byteAddr+3) 0
          let bits := b0.toUInt32 ||| (b1.toUInt32 <<< 8) |||
                      (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
          .ok { s with stack := rest.push (Int.ofNat bits.toNat) }
  | .i32Store offset =>
      match stackPop s.stack with
      | none => .error "i32Store: stack underflow (value)"
      | some (v, rest1) =>
        match stackPop rest1 with
        | none => .error "i32Store: stack underflow (addr)"
        | some (addr, rest2) =>
          let byteAddr := (addr + Int.ofNat offset).toNat
          if byteAddr + 3 >= s.memory.size then
            .error s!"i32Store: address {byteAddr} out of bounds"
          else
            let bits := v.toNat.toUInt32
            let mem := s.memory
              |>.set! byteAddr (UInt8.ofNat (bits &&& 0xFF).toNat)
              |>.set! (byteAddr+1) (UInt8.ofNat ((bits >>> 8) &&& 0xFF).toNat)
              |>.set! (byteAddr+2) (UInt8.ofNat ((bits >>> 16) &&& 0xFF).toNat)
              |>.set! (byteAddr+3) (UInt8.ofNat ((bits >>> 24) &&& 0xFF).toNat)
            .ok { s with stack := rest2, memory := mem }
  -- Control flow (handled by runner; step is a no-op for these)
  | .blockBegin _ | .loopBegin _ | .blockEnd => .ok s
  | .br _ | .brIf _ => .ok s
  | .call _ => .ok s
  | .ret => .ok { s with halted := true }
  | .nop => .ok s
  | .unreachable => .error "unreachable"

/-- Run the native VM interpreter on a function body.
    Simple linear execution without branch resolution. -/
def VMState.run (fn : WasmFunc) (args : Array Int := #[]) : Except String VMState :=
  let nLoc := fn.nLocals
  let initLocals := args ++ Array.replicate (if nLoc > args.size then nLoc - args.size else 0) 0
  let initState : VMState := {
    stack  := #[]
    locals := initLocals
    memory := Array.replicate (64 * 1024) 0
    pc     := 0
  }
  fn.body.foldl (fun stateExcept instr =>
    match stateExcept with
    | .error e => .error e
    | .ok s    =>
      if s.halted then .ok s
      else VMState.step s instr)
    (.ok initState)

/-! ## Transformer execution engine -/

/-- Argmax over a Float32 array. -/
private def argmax (xs : Array Float32) : Nat := Id.run do
  let mut best := 0
  let mut bestVal : Float32 := if xs.isEmpty then 0.0 else xs[0]!
  for i in [:xs.size] do
    if xs[i]! > bestVal then
      best := i
      bestVal := xs[i]!
  pure best

/-- Full transformer state during autoregressive generation. -/
structure TransformerState (dims : LLMDims) where
  caches    : Array LayerCache
  generated : Array Nat := #[]
  deriving Repr, Inhabited

/-- Initialise a fresh transformer state. -/
def TransformerState.init (dims : LLMDims) : TransformerState dims := {
  caches    := Array.replicate dims.layers (LayerCache.empty dims.heads)
  generated := #[]
}

/-- Run one decoding step of the LLM Computer.

    1. Look up token in embedding table
    2. Run all transformer layers (KV cache update)
    3. Project to logits → argmax → next token -/
def transformerStep {dims : LLMDims}
    (weights : LLMComputerWeights dims)
    (inputTok : Nat)
    (state : TransformerState dims)
    : TensorM (Nat × TransformerState dims) := do
  -- ① Build one-hot for inputTok → embed
  let oneHotData : Array Float32 := Id.run do
    let mut v := Array.replicate dims.vocabSize 0.0
    if inputTok < dims.vocabSize then v := v.set! inputTok 1.0
    pure v
  let ohU ← UOp.vconstF32 oneHotData
  let ohBase : StaticTensor [oneHotData.size] .float32 .CPU := StaticTensor.ofUOp ohU
  let ohVec ← reshapeUnsafe ohBase [1, dims.vocabSize]
  let hiddenRow ← matmul ohVec weights.embed   -- [1, dModel]
  let hidden0   ← reshapeUnsafe hiddenRow [dims.dModel]

  -- ② Run transformer layers
  let mut hidden := hidden0
  let mut caches := state.caches
  for l in [:dims.layers] do
    let lw  := weights.layers.getD l (← buildLayerWeights dims l)
    let lc  := caches.getD l (LayerCache.empty dims.heads)
    let (hidden', lc') ← layerStep dims.heads dims.headDim hidden lw lc
    hidden  := hidden'
    caches  := caches.set! l lc'

  -- ③ Project to logits and argmax
  let hiddenRow2 ← reshapeUnsafe hidden [1, dims.dModel]
  let logitsRow  ← matmul hiddenRow2 weights.unembed   -- [1, vocabSize]
  let logitsVec  ← reshapeUnsafe logitsRow [dims.vocabSize]
  let logitsArr  := tensorToArr logitsVec
  let nextTok    := argmax logitsArr

  let newState : TransformerState dims := {
    caches    := caches
    generated := state.generated.push nextTok
  }
  pure (nextTok, newState)

/-! ## Full program execution -/

/-- Run the LLM Computer on a tokenised program until HALT or `maxSteps`. -/
def runTransformer {dims : LLMDims}
    (weights : LLMComputerWeights dims)
    (program : Array Nat)
    (maxSteps : Nat := 1024)
    : TensorM (Array Nat) := do
  let mut state := TransformerState.init dims
  -- Feed program tokens as prompt
  for tok in program do
    let (_, state') ← transformerStep weights tok state
    state := state'
  -- Generate output until HALT
  let mut outputToks : Array Nat := #[]
  let mut lastTok := program.back?.getD TK_HALT
  let mut steps := 0
  while lastTok != TK_HALT && steps < maxSteps do
    let (nextTok, state') ← transformerStep weights lastTok state
    state := state'
    outputToks := outputToks.push nextTok
    lastTok := nextTok
    steps := steps + 1
  pure outputToks

/-- Run the native VM and return the top-of-stack result. -/
def runNative (m : WasmModule) (args : Array Int := #[]) : Except String Int :=
  let fn := m.funcs.getD m.entryPoint (default : WasmFunc)
  match VMState.run fn args with
  | .error e => .error e
  | .ok s    =>
    match s.stack.back? with
    | some v => .ok v
    | none   => .error "runNative: empty stack on exit"

end LLMComputer.Execute
