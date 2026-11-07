import Float64
import TinyGrad4

/-!
# LLMComputer.Types

Core type definitions for the LLM Computer — an architecture where transformer weights
analytically encode a WebAssembly virtual machine, so program execution *is* the forward pass.

Based on Percepta AI's "Constructing an LLM-Computer" (March 2026):
- 7 transformer layers
- d_model = 36
- 18 attention heads × 2 dims/head
- HullKVCache for O(k + log n) decoding instead of O(n²)
-/

namespace LLMComputer

open TinyGrad4

/-! ## Architecture Dimensions -/

/-- The architectural dimensions of an LLM Computer.
    Percepta's reference config: 7 layers, d=36, 18 heads of 2 dims each. -/
structure LLMDims where
  layers    : Nat := 7
  dModel    : Nat := 36
  heads     : Nat := 18
  headDim   : Nat := 2    -- must equal dModel / heads
  ffnMult   : Nat := 4    -- feed-forward hidden = ffnMult * dModel
  vocabSize : Nat := 512  -- 0–255: data bytes; 256–511: opcode tokens
  deriving Repr, Inhabited

/-- Default Percepta-style dimensions -/
def LLMDims.default : LLMDims := {}

/-- Validate that head dimensions are consistent -/
def LLMDims.valid (d : LLMDims) : Bool :=
  d.heads * d.headDim == d.dModel

/-! ## WebAssembly Instruction Set (Flattened) -/

/-- Simplified, flat WebAssembly instruction set.
    We flatten blocks/loops into label-based jumps (like a bytecode VM)
    so the sequence is a plain `Array WasmInstr` — easy to tokenise. -/
inductive WasmInstr where
  -- Constants
  | i32Const  (v : Int)       -- push i32 constant onto stack
  -- Arithmetic
  | i32Add                    -- pop a, b; push a + b
  | i32Sub                    -- pop a, b; push a - b
  | i32Mul                    -- pop a, b; push a * b
  | i32DivS                   -- pop a, b; push a / b (signed)
  | i32RemS                   -- pop a, b; push a % b (signed)
  -- Comparison (result: 0 or 1)
  | i32Eq                     -- pop a, b; push (a == b ? 1 : 0)
  | i32Ne                     -- pop a, b; push (a != b ? 1 : 0)
  | i32LtS                    -- pop a, b; push (a < b ? 1 : 0)
  | i32GtS                    -- pop a, b; push (a > b ? 1 : 0)
  | i32LeS
  | i32GeS
  -- Bitwise
  | i32And | i32Or | i32Xor
  | i32Shl | i32ShrS
  -- Stack ops
  | drop                      -- discard top of stack
  | dup                       -- duplicate top of stack (non-WASM but useful)
  -- Local variables
  | localGet (idx : Nat)      -- push locals[idx]
  | localSet (idx : Nat)      -- pop; store to locals[idx]
  | localTee (idx : Nat)      -- localSet without consuming (peek+store)
  -- Memory
  | i32Load  (offset : Nat)   -- pop addr; push mem[addr+offset]
  | i32Store (offset : Nat)   -- pop val, addr; mem[addr+offset] = val
  -- Control flow (label-based, resolved at compile time)
  | blockBegin (label : Nat)  -- begin a block with given label id
  | loopBegin  (label : Nat)  -- begin a loop with given label id
  | blockEnd                  -- end current block/loop
  | br    (label : Nat)       -- unconditional branch to label
  | brIf  (label : Nat)       -- conditional branch (pop condition)
  | call  (fnIdx : Nat)       -- call function by index
  | ret                       -- return (top of stack = result)
  | nop                       -- no operation
  | unreachable               -- trap
  deriving Repr, Inhabited, BEq, Hashable

/-- A complete WebAssembly function (flat instruction sequence) -/
structure WasmFunc where
  name    : String             := "anonymous"
  nLocals : Nat                := 0   -- number of mutable local variables
  nParams : Nat                := 0   -- parameters (subset of locals)
  body    : Array WasmInstr    := #[]
  deriving Repr, Inhabited

/-- A WebAssembly module — the unit of compilation -/
structure WasmModule where
  funcs      : Array WasmFunc  := #[]
  memSize    : Nat             := 1   -- memory pages (64 KB each)
  entryPoint : Nat             := 0   -- index of entry function
  deriving Repr, Inhabited

/-! ## VM Execution Tokens

The transformer works token-by-token.  Each token represents one "event"
in the VM's execution trace — either a pushed value, an opcode being executed,
or a local variable binding change.  The token vocabulary encodes this trace.
-/

/-- Token categories in the LLM Computer vocabulary -/
inductive VMToken where
  | data    (v : Int)          -- integer value on the virtual stack
  | opcode  (id : Nat)         -- opcode token (id into vocab 256–511)
  | local   (idx : Int)        -- local variable reference token
  | control (label : Nat)      -- control-flow label token
  | halt                       -- program finished, emit result
  | padding                    -- padding / beginning-of-sequence
  deriving Repr, Inhabited, BEq

/-! ## Transformer Layer Weights

The weights of an LLM Computer are *analytically constructed* from the WasmModule.
They are not learned by gradient descent — they are hand-compiled so that the
transformer's forward pass exactly simulates the VM.
-/

/-- Weights for one transformer layer.
    All matrices live on CPU for now; Metal/CUDA can be added trivially. -/
structure LayerWeights (dModel ffnHidden : Nat) where
  -- Multi-head self-attention projections
  wQ  : Matrix dModel dModel .float32      -- query
  wK  : Matrix dModel dModel .float32      -- key
  wV  : Matrix dModel dModel .float32      -- value
  wO  : Matrix dModel dModel .float32      -- output
  -- Feed-forward network (GELU activation between ff1 and ff2)
  ff1 : Matrix dModel ffnHidden .float32
  ff2 : Matrix ffnHidden dModel .float32
  -- Layer-norm scale parameters (bias omitted for simplicity)
  lnAttn : Vector dModel .float32          -- pre-attention LN scale
  lnFFN  : Vector dModel .float32          -- pre-FFN LN scale
  deriving Repr

/-- The complete analytically compiled weight set for an LLM Computer -/
structure LLMComputerWeights (dims : LLMDims) where
  -- Token embedding table [vocabSize, dModel]
  embed   : Matrix dims.vocabSize dims.dModel .float32
  -- Per-layer weights
  layers  : Array (LayerWeights dims.dModel (dims.dModel * dims.ffnMult))
  -- Final unembedding (logit head) [dModel, vocabSize]
  unembed : Matrix dims.dModel dims.vocabSize .float32

/-! ## HullKVCache

Percepta's key efficiency contribution: instead of O(n²) full attention,
maintain only the "convex hull" of past key vectors.  Future queries can
be answered accurately using only the hull vertices.

Intuition: if K_i is in the convex hull of {K_j : j ≠ i}, then the
attention score q·K_i is always dominated by some convex combination of
{q·K_j}, so K_i (and its V_i) can be pruned without changing the output.

For program execution, the "hull" turns out to be very small (O(log n) entries
for arithmetic-only traces) because the key vectors live on a low-dimensional
manifold determined by the stack depth.
-/

/-- One entry in the hull KV cache -/
structure KVEntry (headDim : Nat) where
  key   : Vector headDim .float32   -- the key vector for this position
  value : Vector headDim .float32   -- the corresponding value vector
  pos   : Nat                       -- original sequence position
  deriving Repr, Inhabited

/-- Per-head hull KV cache (grows during decoding, prunes dominated entries) -/
structure HullKVCache (headDim : Nat) where
  /-- Active hull entries (hull vertices in K-space) -/
  hull    : Array (KVEntry headDim)
  /-- Max entries before forced pruning (capacity bound) -/
  maxSize : Nat := 256
  deriving Repr, Inhabited

/-- Full KV cache for all layers and all heads -/
structure LLMComputerCache (dims : LLMDims) where
  /-- caches.(layer).(head) -/
  caches : Array (Array (HullKVCache dims.headDim))
  deriving Repr, Inhabited

/-- VM execution state — what gets threaded through the transformer steps -/
structure VMState where
  stack    : Array Int       -- data stack (grows right)
  locals   : Array Int       -- local variable bindings
  memory   : Array UInt8     -- flat byte memory
  pc       : Nat             -- program counter (instruction index)
  halted   : Bool := false   -- true when ret/unreachable hit
  deriving Repr, Inhabited

end LLMComputer
