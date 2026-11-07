import LLMComputer.Types

/-!
# LLMComputer.Vocab

Token vocabulary for the LLM Computer.

## Layout (512 tokens total)

| Range     | Meaning                                      |
|-----------|----------------------------------------------|
| 0 – 255   | Signed integer data values (–128 … 127, then 128 … 255 as positive bytes) |
| 256       | `i32.const` (next token = value)             |
| 257       | `i32.add`                                    |
| 258       | `i32.sub`                                    |
| 259       | `i32.mul`                                    |
| 260       | `i32.divs`                                   |
| 261       | `i32.rems`                                   |
| 262       | `i32.eq`                                     |
| 263       | `i32.ne`                                     |
| 264       | `i32.lts`                                    |
| 265       | `i32.gts`                                    |
| 266       | `i32.les`                                    |
| 267       | `i32.ges`                                    |
| 268       | `i32.and`                                    |
| 269       | `i32.or`                                     |
| 270       | `i32.xor`                                    |
| 271       | `i32.shl`                                    |
| 272       | `i32.shrs`                                   |
| 273       | `drop`                                       |
| 274       | `dup`                                        |
| 275       | `local.get` (next token = local index)       |
| 276       | `local.set` (next token = local index)       |
| 277       | `local.tee` (next token = local index)       |
| 278       | `i32.load`                                   |
| 279       | `i32.store`                                  |
| 280       | `block.begin`                                |
| 281       | `loop.begin`                                 |
| 282       | `block.end`                                  |
| 283       | `br`       (next token = label)              |
| 284       | `br.if`    (next token = label)              |
| 285       | `call`     (next token = fn index)           |
| 286       | `ret`                                        |
| 287       | `nop`                                        |
| 288       | `unreachable`                                |
| 289       | `halt`     (end-of-output sentinel)          |
| 290 – 383 | Local-variable index tokens (0–93)           |
| 384 – 447 | Label tokens (0–63)                          |
| 448 – 511 | Function-index tokens (0–63)                 |
-/

namespace LLMComputer.Vocab

open LLMComputer

/-! ## Token ID constants -/

def TK_I32CONST   : Nat := 256
def TK_I32ADD     : Nat := 257
def TK_I32SUB     : Nat := 258
def TK_I32MUL     : Nat := 259
def TK_I32DIVS    : Nat := 260
def TK_I32REMS    : Nat := 261
def TK_I32EQ      : Nat := 262
def TK_I32NE      : Nat := 263
def TK_I32LTS     : Nat := 264
def TK_I32GTS     : Nat := 265
def TK_I32LES     : Nat := 266
def TK_I32GES     : Nat := 267
def TK_I32AND     : Nat := 268
def TK_I32OR      : Nat := 269
def TK_I32XOR     : Nat := 270
def TK_I32SHL     : Nat := 271
def TK_I32SHRS    : Nat := 272
def TK_DROP       : Nat := 273
def TK_DUP        : Nat := 274
def TK_LOCALGET   : Nat := 275
def TK_LOCALSET   : Nat := 276
def TK_LOCALTEE   : Nat := 277
def TK_I32LOAD    : Nat := 278
def TK_I32STORE   : Nat := 279
def TK_BLOCKBEGIN : Nat := 280
def TK_LOOPBEGIN  : Nat := 281
def TK_BLOCKEND   : Nat := 282
def TK_BR         : Nat := 283
def TK_BRIF       : Nat := 284
def TK_CALL       : Nat := 285
def TK_RET        : Nat := 286
def TK_NOP        : Nat := 287
def TK_UNREACHABLE: Nat := 288
def TK_HALT       : Nat := 289

def TK_LOCAL_BASE : Nat := 290
def TK_LABEL_BASE : Nat := 384
def TK_FN_BASE    : Nat := 448
def VOCAB_SIZE    : Nat := 512

/-! ## Encoding: WasmInstr → Token IDs -/

/-- Encode a signed integer value as a token ID.
    Values –128..127 map to tokens 0..255. -/
def encodeValue (v : Int) : Nat :=
  let clamped := v.toNat % 256   -- wrap to byte range
  clamped

/-- Decode a token ID back to a signed integer value. -/
def decodeValue (tok : Nat) : Int :=
  if tok < 128 then tok else (tok : Int) - 256

/-- Encode a `WasmInstr` as a sequence of token IDs.
    Instructions with immediate arguments (i32Const, localGet, …)
    emit two tokens: the opcode token followed by the argument token. -/
def encodeInstr (instr : WasmInstr) : Array Nat :=
  match instr with
  | .i32Const v       => #[TK_I32CONST, encodeValue v]
  | .i32Add           => #[TK_I32ADD]
  | .i32Sub           => #[TK_I32SUB]
  | .i32Mul           => #[TK_I32MUL]
  | .i32DivS          => #[TK_I32DIVS]
  | .i32RemS          => #[TK_I32REMS]
  | .i32Eq            => #[TK_I32EQ]
  | .i32Ne            => #[TK_I32NE]
  | .i32LtS           => #[TK_I32LTS]
  | .i32GtS           => #[TK_I32GTS]
  | .i32LeS           => #[TK_I32LES]
  | .i32GeS           => #[TK_I32GES]
  | .i32And           => #[TK_I32AND]
  | .i32Or            => #[TK_I32OR]
  | .i32Xor           => #[TK_I32XOR]
  | .i32Shl           => #[TK_I32SHL]
  | .i32ShrS          => #[TK_I32SHRS]
  | .drop             => #[TK_DROP]
  | .dup              => #[TK_DUP]
  | .localGet idx     => #[TK_LOCALGET,   TK_LOCAL_BASE + idx]
  | .localSet idx     => #[TK_LOCALSET,   TK_LOCAL_BASE + idx]
  | .localTee idx     => #[TK_LOCALTEE,   TK_LOCAL_BASE + idx]
  | .i32Load offset   => #[TK_I32LOAD,    TK_LABEL_BASE + offset]
  | .i32Store offset  => #[TK_I32STORE,   TK_LABEL_BASE + offset]
  | .blockBegin label => #[TK_BLOCKBEGIN, TK_LABEL_BASE + label]
  | .loopBegin label  => #[TK_LOOPBEGIN,  TK_LABEL_BASE + label]
  | .blockEnd         => #[TK_BLOCKEND]
  | .br label         => #[TK_BR,         TK_LABEL_BASE + label]
  | .brIf label       => #[TK_BRIF,       TK_LABEL_BASE + label]
  | .call fnIdx       => #[TK_CALL,       TK_FN_BASE    + fnIdx]
  | .ret              => #[TK_RET]
  | .nop              => #[TK_NOP]
  | .unreachable      => #[TK_UNREACHABLE]

/-- Tokenise an entire function body. -/
def encodeFunc (fn : WasmFunc) : Array Nat :=
  fn.body.foldl (fun acc instr => acc ++ encodeInstr instr) #[]

/-- Tokenise a full module (all functions, entry-point first). -/
def encodeModule (m : WasmModule) : Array Nat :=
  let entry := m.funcs.getD m.entryPoint (default : WasmFunc)
  encodeFunc entry

/-! ## Embedding vectors

Each token maps to a 36-dim embedding vector with a specific structure:

  dims  0 –  7 : value bits (8 bits of the integer payload)
  dims  8 – 15 : opcode category one-hot (8 categories)
  dims 16 – 23 : argument field (local index, label, …)
  dims 24 – 35 : position / stack-depth signal (12 dims, set by the transformer)

The embedding for token `t` is computed analytically here so the compiler
can construct the full embedding matrix row by row.
-/

/-- 8 opcode categories used in the embedding one-hot -/
inductive OpcodeCategory where
  | const     -- i32.const
  | arith     -- add, sub, mul, div, rem
  | compare   -- eq, ne, lt, gt, le, ge
  | bitwise   -- and, or, xor, shl, shr
  | stack     -- drop, dup
  | local     -- local.get / set / tee
  | memory    -- load / store
  | control   -- br, brif, call, ret, nop, block, loop
  deriving Repr, Inhabited, BEq

/-- Map a token ID to its opcode category (for data tokens: use `const`) -/
def tokenCategory (tok : Nat) : OpcodeCategory :=
  if tok < 256                          then .const
  else if tok == TK_I32CONST            then .const
  else if tok >= TK_I32ADD && tok <= TK_I32REMS  then .arith
  else if tok >= TK_I32EQ  && tok <= TK_I32GES   then .compare
  else if tok >= TK_I32AND && tok <= TK_I32SHRS  then .bitwise
  else if tok == TK_DROP || tok == TK_DUP         then .stack
  else if tok >= TK_LOCALGET && tok <= TK_LOCALTEE then .local
  else if tok == TK_I32LOAD || tok == TK_I32STORE  then .memory
  else .control

/-- Opcode category as a one-hot index (0–7) -/
def categoryIndex : OpcodeCategory → Nat
  | .const   => 0
  | .arith   => 1
  | .compare => 2
  | .bitwise => 3
  | .stack   => 4
  | .local   => 5
  | .memory  => 6
  | .control => 7

/-- Sub-opcode within a category (for fine-grained attention routing) -/
def subOpcode (tok : Nat) : Nat :=
  if tok < 256      then tok % 8                          -- value low bits
  else if tok < 290 then (tok - 256) % 8                 -- opcode within category
  else if tok < 384 then (tok - TK_LOCAL_BASE) % 8       -- local index mod 8
  else if tok < 448 then (tok - TK_LABEL_BASE) % 8       -- label mod 8
  else (tok - TK_FN_BASE) % 8                            -- fn index mod 8

/-- Build the 36-dimensional embedding vector for token `tok`.
    Returns a plain `Array Float32` (36 elements). -/
def tokenEmbedding (tok : Nat) : Array Float32 :=
  let v := Id.run do
    let mut e : Array Float32 := Array.replicate 36 0.0
    -- dims 0–7: value bits
    let valueBits : Nat :=
      if tok < 256 then tok
      else 0
    for i in [:8] do
      e := e.set! i (if (valueBits >>> i) % 2 == 1 then 1.0 else 0.0)
    -- dims 8–15: opcode category one-hot
    let catIdx := categoryIndex (tokenCategory tok)
    e := e.set! (8 + catIdx) 1.0
    -- dims 16–23: sub-opcode bits
    let sub := subOpcode tok
    for i in [:8] do
      e := e.set! (16 + i) (if (sub >>> i) % 2 == 1 then 1.0 else 0.0)
    -- dims 24–35: zero at construction time; transformer layers will fill these
    pure e
  v

/-- Build the full embedding matrix [vocabSize × dModel] as a flat Array -/
def buildEmbeddingData (vocabSize dModel : Nat) : Array Float32 :=
  let embDim := min dModel 36   -- clip if dModel < 36 (shouldn't happen normally)
  Id.run do
    let mut data : Array Float32 := Array.replicate (vocabSize * dModel) 0.0
    for tok in [:vocabSize] do
      let emb := tokenEmbedding tok
      for d in [:embDim] do
        data := data.set! (tok * dModel + d) (emb.getD d 0.0)
    pure data

end LLMComputer.Vocab
