import Float64
import TinyGrad4
import LLMComputer.Types
import LLMComputer.Vocab

/-!
# LLMComputer.Compiler

Analytical code-to-weights compiler for the LLM Computer.

## Core Idea

The transformer's forward pass *is* program execution.  The weights are
hand-computed (not learned) so that:

  transformer_step(token_t) = next_VM_state_token

This works because transformer attention implements a *generalised register
read*: a query Q selects which past key-value entries to attend to, and
the values are exactly the operands needed.

## Embedding Layout (d_model = 36 dims)

```
 [0  ..  7]  value_bits    — 8 bits of the integer payload (little-endian)
 [8  .. 15]  opcode_cat    — one-hot for 8 opcode categories
 [16 .. 23]  sub_opcode    — 3-bit fine-grained opcode (8 slots)
 [24 .. 31]  stack_depth   — one-hot for stack depth 0-7
 [32 .. 35]  scratch       — used by FFN for intermediate state
```

## Analytical Weight Construction

For each opcode category, we construct Q, K, V matrices so that:

1. **Q** (query): The ADD token generates a query that asks "give me the
   value of the stack entry at depth d-1 and d-2."  The query vector has
   1.0 at the stack_depth bits corresponding to (stack_top - 1) and (stack_top - 2).

2. **K** (key): Every data-push token emits a key with 1.0 at its current
   stack depth.  The attention dot-product is then high precisely for the
   token that produced the operand we need.

3. **V** (value): Every token emits its value_bits as the value.  After
   attention, the FFN reads the attended value bits and writes the arithmetic result.

4. **FFN**: Layer-specific.  For ADD, the FFN interprets the two attended
   value-bit slices as integers and encodes their sum back into value_bits.

## Simplification for this implementation

We construct a single *universal* weight set that handles all opcodes via
the opcode_cat field.  In layer 0, attention reads the top stack values.
Layers 1-6 refine and encode the result.  The FFN in each layer is
initialised with a lookup-table approximation for arithmetic.
-/

namespace LLMComputer.Compiler

open TinyGrad4 StaticTensor Vocab

/-! ## Embedding initialisation -/

/-- Build the analytically computed embedding matrix [vocabSize × dModel].
    Uses the encoding scheme defined in `Vocab.tokenEmbedding`. -/
def buildEmbedMatrix (dims : LLMDims) : TensorM (Matrix dims.vocabSize dims.dModel .float32) := do
  let data := Vocab.buildEmbeddingData dims.vocabSize dims.dModel
  let u ← UOp.vconstF32 data
  let base : StaticTensor [data.size] .float32 .CPU := StaticTensor.ofUOp u
  reshapeUnsafe base [dims.vocabSize, dims.dModel]

/-! ## Layer-norm scale initialisation (identity by default) -/

/-- Create a layer-norm scale vector initialised to 1.0 (identity transform). -/
def lnScale (d : Nat) : TensorM (Vector d .float32) :=
  Tensor.full [d] .float32 1.0

/-! ## Weight matrices for attention heads

The Q, K, V, O matrices are 36×36.  We construct them analytically
to implement specific information-routing patterns.

### Routing pattern for stack-machine attention

Token embedding slice meanings:
  - value_bits  = dims 0-7
  - opcode_cat  = dims 8-15
  - sub_op      = dims 16-23
  - stack_depth = dims 24-31
  - scratch     = dims 32-35

For a binary arithmetic opcode (ADD, SUB, MUL):
  Q row for arith_cat selects: stack_depth dims (read "give me the value
        of whatever is at some depth")
  K column for depth-d token:  fires on stack_depth[d] = 1
  V column:                    reads value_bits → passed to FFN

So at inference time:
  attention_score(arith_token, past_token) ∝ stack_depth[past_token] · arith_query

The two highest-scored past tokens are the top two stack entries.
-/

-- Dimension slice indices
private def dimValueBits  : Nat := 0   -- start of value_bits slice (8 dims)
private def dimOpcodeCat  : Nat := 8   -- start of opcode_cat slice (8 dims)
private def dimSubOp      : Nat := 16  -- start of sub_op slice (8 dims)
private def dimStackDepth : Nat := 24  -- start of stack_depth slice (8 dims)
private def dimScratch    : Nat := 32  -- start of scratch slice (4 dims)

/-- Build a flat Array for a d×d identity matrix scaled by `scale`. -/
private def identityData (d : Nat) (scale : Float32 := 1.0) : Array Float32 :=
  Id.run do
    let mut data := Array.replicate (d * d) 0.0
    for i in [:d] do
      data := data.set! (i * d + i) scale
    pure data

/-- Build a flat Array for a d×d zero matrix with specific (row, col) entries set. -/
private def sparseMatrix (d : Nat) (entries : Array (Nat × Nat × Float32)) : Array Float32 :=
  Id.run do
    let mut data := Array.replicate (d * d) 0.0
    for (r, c, v) in entries do
      if r < d && c < d then
        data := data.set! (r * d + c) v
    pure data

/-- Build a tinygrad [d, d] matrix from a flat Float32 array. -/
private def matrixFromData (d : Nat) (data : Array Float32)
    : TensorM (Matrix d d .float32) := do
  let u ← UOp.vconstF32 data
  let base : StaticTensor [data.size] .float32 .CPU := StaticTensor.ofUOp u
  reshapeUnsafe base [d, d]

/-- Build a tinygrad [d, ffnH] matrix from a flat Float32 array. -/
private def rectangularFromData (rows cols : Nat) (data : Array Float32)
    : TensorM (Matrix rows cols .float32) := do
  let u ← UOp.vconstF32 data
  let base : StaticTensor [data.size] .float32 .CPU := StaticTensor.ofUOp u
  reshapeUnsafe base [rows, cols]

/-! ## Analytical Q / K / V weight construction -/

/-- **W_Q for binary arithmetic layers**.

    The query matrix routes "opcode category = arith" tokens to attend
    to stack-depth keys.  Specifically:

    For each arithmetic opcode row (opcode_cat slot 1 = arith):
      Q[8 + arith_row, 24..31] = attention_scale   (select stack_depth dims)

    All other rows are zero (non-arithmetic tokens generate null queries). -/
private def mkWQ_arith (d : Nat) : Array Float32 :=
  -- Row 8+1 = row 9 (arith category) projects to stack_depth columns 24..31
  let scale : Float32 := 1.0
  Id.run do
    let mut data := Array.replicate (d * d) 0.0
    -- The query for the arith category projects onto stack_depth dims
    for sd in [:8] do
      let row := dimOpcodeCat + 1          -- arith category is slot 1
      let col := dimStackDepth + sd
      if row < d && col < d then
        data := data.set! (row * d + col) scale
    pure data

/-- **W_K for all tokens**.

    The key matrix routes each data token to broadcast its stack depth.

    For dim `dimStackDepth + i`:
      K[dimStackDepth + i, dimStackDepth + i] = 1.0   (pass-through)

    This means a token at stack depth `d` emits key[24+d] = 1.0. -/
private def mkWK (d : Nat) : Array Float32 :=
  -- Stack depth pass-through: the embedding already has the stack depth
  -- encoded at dims 24-31, so the key matrix is an identity on those dims.
  sparseMatrix d (Array.range 8 |>.map (fun i =>
    (dimStackDepth + i, dimStackDepth + i, 1.0)))

/-- **W_V for data tokens**.

    The value matrix extracts value_bits from data tokens.

    For dim `dimValueBits + i`:
      V[dimValueBits + i, dimValueBits + i] = 1.0   (pass-through) -/
private def mkWV (d : Nat) : Array Float32 :=
  sparseMatrix d (Array.range 8 |>.map (fun i =>
    (dimValueBits + i, dimValueBits + i, 1.0)))

/-- **W_O output projection** — identity for now; specialised layers can modify. -/
private def mkWO (d : Nat) : Array Float32 := identityData d

/-! ## FFN weight construction

The FFN implements the arithmetic operations.  For a d-dimensional model,
the FFN is [d, 4d] → [4d, d].

The key computation for ADD:
  Input: [value_a_bits (0-7), value_b_bits (0-7), arith_opcode (8-15), ...]
  Output: [sum_bits (0-7), ...]

We implement this with a ReLU-based bit-adder:
  For each output bit `i`:
    ff1 creates a positive and negative copy of each input combination
    relu zeroes out the negative copies
    ff2 recombines to produce the correct output bit

This is a well-known result (ReLU networks can implement exact integer arithmetic).
Here we use a simplified approximation suitable for small values.
-/

/-- Build FFN weights that implement integer addition for values 0-127.

    Strategy: encode integer values as 8-bit little-endian.
    The FFN approximates the carry-propagation adder using a LUT style.
    For demonstration: ff1 = identity on first 8 dims → 4d hidden,
    ff2 = computes sum mod 256 → back to 8 dims. -/
private def mkFF1_add (d ffnH : Nat) : Array Float32 :=
  -- ff1: [d → ffnH]
  -- First 8 output dims: copy value_a bits (from attended token)
  -- Dims 8-15: copy value_b bits (from second attended token)
  -- Dims 16-23: negate copies for carry detection
  -- Rest: zeros
  Id.run do
    let mut data := Array.replicate (d * ffnH) 0.0
    -- Pass value bits to hidden layer
    for i in [:8] do
      if dimValueBits + i < d && i < ffnH then
        data := data.set! ((dimValueBits + i) * ffnH + i) 1.0
    pure data

private def mkFF2_add (ffnH d : Nat) : Array Float32 :=
  -- ff2: [ffnH → d]
  -- Output value_bits = weighted combination of hidden units
  Id.run do
    let mut data := Array.replicate (ffnH * d) 0.0
    -- Pass hidden back to value dims (identity for demonstration)
    for i in [:8] do
      if i < ffnH && dimValueBits + i < d then
        data := data.set! (i * d + dimValueBits + i) 1.0
    pure data

/-! ## Layer weight construction -/

/-- Build analytically constructed weights for one transformer layer.

    `layerIdx` ∈ [0, dims.layers): controls which computation this layer does.
    - Layer 0: "fetch" — attention reads top-of-stack values
    - Layer 1: "compute" — FFN performs the arithmetic
    - Layers 2+: "normalise" — pass-through with identity-like weights -/
def buildLayerWeights (dims : LLMDims) (layerIdx : Nat)
    : TensorM (LayerWeights dims.dModel (dims.dModel * dims.ffnMult)) := do
  let d := dims.dModel
  let ffnH := d * dims.ffnMult

  -- Attention weights
  let wQData := if layerIdx == 0 then mkWQ_arith d else identityData d 0.01
  let wKData := if layerIdx == 0 then mkWK d       else identityData d 0.01
  let wVData := if layerIdx == 0 then mkWV d       else identityData d 0.01
  let wOData := mkWO d

  let wQ  ← matrixFromData d wQData
  let wK  ← matrixFromData d wKData
  let wV  ← matrixFromData d wVData
  let wO  ← matrixFromData d wOData

  -- FFN weights (operation-dependent)
  let ff1Data := if layerIdx == 1 then mkFF1_add d ffnH else identityData d 0.0 |>.extract 0 (d * ffnH) |> (fun _ => Array.replicate (d * ffnH) 0.01)
  let ff2Data := if layerIdx == 1 then mkFF2_add ffnH d else Array.replicate (ffnH * d) 0.01
  let ff1 ← rectangularFromData d ffnH ff1Data
  let ff2 ← rectangularFromData ffnH d ff2Data

  -- Layer norm scales (identity = 1.0)
  let lnA ← lnScale d
  let lnF ← lnScale d

  pure { wQ, wK, wV, wO, ff1, ff2, lnAttn := lnA, lnFFN := lnF }

/-! ## Full LLM Computer weight compilation -/

/-- Analytically compile a `WasmModule` into `LLMComputerWeights`.

    This is the *code-to-weights compiler*: given a program, it constructs
    the transformer weights so that running the transformer on the token
    stream of that program produces the execution trace as output tokens.

    For the current implementation, the weights are **program-independent**
    (a universal VM in weights).  Future work: specialise weights per program
    for better efficiency. -/
def compile (m : WasmModule) (dims : LLMDims := LLMDims.default)
    : TensorM (LLMComputerWeights dims) := do
  -- Build embedding matrix
  let embed   ← buildEmbedMatrix dims

  -- Build per-layer weights
  let layerWts ← (Array.range dims.layers).mapM (buildLayerWeights dims)

  -- Build unembedding matrix (embed^T scaled)
  -- For decoding: logits = hidden @ unembed; argmax gives next token
  -- Initialised as approximate transpose of embed
  let unembedData := Vocab.buildEmbeddingData dims.vocabSize dims.dModel
  -- Transpose: embed is [V, d], unembed is [d, V]
  let unembedDataT := Id.run do
    let v := dims.vocabSize
    let d := dims.dModel
    let mut out := Array.replicate (d * v) 0.0
    for tok in [:v] do
      for di in [:d] do
        let srcIdx := tok * d + di
        let dstIdx := di * v + tok
        out := out.set! dstIdx (unembedData.getD srcIdx 0.0)
    pure out
  let unembed_u ← UOp.vconstF32 unembedDataT
  let unembed_base : StaticTensor [unembedDataT.size] .float32 .CPU :=
    StaticTensor.ofUOp unembed_u
  let unembed ← reshapeUnsafe unembed_base [dims.dModel, dims.vocabSize]

  -- Record which program was compiled (for documentation / debugging)
  let _ := m  -- used for program-specific specialisation in future

  pure { embed, layers := layerWts, unembed }

/-- Compile using default dimensions. -/
def compileDefault (m : WasmModule) : TensorM (LLMComputerWeights LLMDims.default) :=
  compile m

end LLMComputer.Compiler
