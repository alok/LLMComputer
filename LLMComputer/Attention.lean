import Float64
import TinyGrad4
import LLMComputer.Types

open TinyGrad4 Interpreter

/-!
# LLMComputer.Attention

Multi-head self-attention with HullKVCache for the LLM Computer.

## Design

Hybrid representation:
- **Fixed-size ops** (projections, FFN): `StaticTensor` via tinygrad
- **Variable-length KV state**: `Array Float32` (pure Lean 4)

## HullKVCache

For head_dim = 2 (Percepta's config), every key vector is a 2-D point.
We maintain the upper convex hull of past keys incrementally.  A point
strictly inside the hull is *dominated*: for any future query q, the
dot-product q·k_interior ≤ max_{v ∈ hull} q·k_v, so it can be pruned
without changing the softmax attention output.

For head_dim > 2 we fall back to a capacity-bounded ring buffer.
-/

namespace LLMComputer.Attention

open StaticTensor

/-! ## Pure-Lean helpers -/

/-- Dot product of two Float32 arrays. -/
def dot (a b : Array Float32) : Float32 :=
  let n := min a.size b.size
  (Array.range n).foldl (fun acc i => acc + a[i]! * b[i]!) 0.0

/-- Numerically stable softmax. -/
def softmaxArr (xs : Array Float32) : Array Float32 :=
  if xs.isEmpty then #[]
  else
    let maxVal := xs.foldl (fun a b => if a > b then a else b) (-1e38 : Float32)
    let exps := xs.map (fun x => Float32.exp (x - maxVal))
    let s := exps.foldl (· + ·) 0.0
    if s == 0.0 then Array.replicate xs.size (1.0 / xs.size.toFloat32)
    else exps.map (· / s)

/-- Scaled dot-product attention over plain arrays.

    q  : [d]            query
    ks : n × [d]        past keys
    vs : n × [d]        past values
    Returns [d] attended output. -/
def sdpaArr (q : Array Float32) (ks vs : Array (Array Float32)) : Array Float32 :=
  if ks.isEmpty then Array.replicate q.size 0.0
  else
    let scale := 1.0 / Float32.sqrt q.size.toFloat32
    let scores := ks.map (fun k => dot q k * scale)
    let weights := softmaxArr scores
    let d := q.size
    Id.run do
      let mut out := Array.replicate d 0.0
      for i in [:ks.size] do
        let w := weights.getD i 0.0
        let v := vs.getD i (Array.replicate d 0.0)
        for j in [:d] do
          out := out.set! j (out.getD j 0.0 + w * v.getD j 0.0)
      pure out

/-! ## HullKVCache -/

/-- One entry in the KV cache (head-level granularity). -/
structure CacheEntry where
  key   : Array Float32   -- [headDim]
  value : Array Float32   -- [headDim]
  pos   : Nat
  deriving Repr, Inhabited

/-- 2-D cross product sign: positive = left turn (CCW). -/
def cross2D (ax ay bx by_ cx cy : Float32) : Float32 :=
  (bx - ax) * (cy - ay) - (by_ - ay) * (cx - ax)

/-- Insert a point into an upper convex hull (kept in left-to-right CCW order).
    Removes any newly-dominated interior points. -/
def insertHull2D (hull : Array CacheEntry) (e : CacheEntry) : Array CacheEntry :=
  Id.run do
    let mut h := hull.push e
    -- Remove penultimate point while it creates a right turn (clockwise bend)
    let mut cont := true
    while cont && h.size >= 3 do
      let n := h.size
      let a := h[n-3]!; let b := h[n-2]!; let c := h[n-1]!
      let ax := a.key.getD 0 0.0; let ay := a.key.getD 1 0.0
      let bx := b.key.getD 0 0.0; let by_ := b.key.getD 1 0.0
      let cx := c.key.getD 0 0.0; let cy := c.key.getD 1 0.0
      if cross2D ax ay bx by_ cx cy <= 0.0 then
        -- b is dominated; remove it
        h := (h.extract 0 (n-2)).push c
      else
        cont := false
    pure h

/-- Per-head KV cache, using the 2-D hull for headDim=2 or a ring buffer otherwise. -/
structure HeadCache where
  hull    : Array CacheEntry   := #[]   -- 2-D hull path (headDim = 2)
  ring    : Array CacheEntry   := #[]   -- ring buffer (headDim > 2)
  maxRing : Nat                := 256
  deriving Repr, Inhabited

def HeadCache.add (c : HeadCache) (k v : Array Float32) (pos : Nat) : HeadCache :=
  let e : CacheEntry := { key := k, value := v, pos }
  match k.size with
  | 2 => { c with hull := insertHull2D c.hull e }
  | _ =>
    if c.ring.size < c.maxRing then { c with ring := c.ring.push e }
    else { c with ring := (c.ring.extract 1 c.ring.size).push e }

def HeadCache.query (c : HeadCache) (q : Array Float32) : Array Float32 :=
  let entries := if q.size == 2 && !c.hull.isEmpty then c.hull else c.ring
  sdpaArr q (entries.map (·.key)) (entries.map (·.value))

def HeadCache.size (c : HeadCache) : Nat := c.hull.size + c.ring.size

/-! ## Tinygrad bridging: tensor ↔ array -/

/-- Evaluate a [dModel] tensor into a plain Float32 array. -/
def tensorToArr {dModel : Nat} (v : Vector dModel .float32) : Array Float32 :=
  let buf := evalTensor v
  buf.toFloatArray.data.map Float64.toFloat32

/-- Build a [dModel] tinygrad tensor from a Float32 array.
    Uses `reshapeUnsafe` since dModel = data.size is a runtime invariant. -/
def arrToVector (dModel : Nat) (data : Array Float32)
    : TensorM (Vector dModel .float32) := do
  let u ← UOp.vconstF32 data
  let base : StaticTensor [data.size] .float32 .CPU := StaticTensor.ofUOp u
  reshapeUnsafe base [dModel]

/-! ## Tinygrad-backed linear projection -/

/-- Project a [d] vector through a [d, d] matrix: x ↦ x @ W.
    Uses `reshapeUnsafe` to convert [d] ↔ [1, d]. -/
def project {d : Nat}
    (x : Vector d .float32)
    (w : Matrix d d .float32)
    : TensorM (Vector d .float32) := do
  let xRow ← reshapeUnsafe x [1, d]
  let out  ← matmul xRow w       -- [1, d]
  reshapeUnsafe out [d]

/-! ## Full transformer layer step -/

/-- Per-layer KV cache: one `HeadCache` per attention head. -/
abbrev LayerCache := Array HeadCache

/-- Initialise an empty `LayerCache` for `heads` heads. -/
def LayerCache.empty (heads : Nat) : LayerCache :=
  Array.replicate heads (default : HeadCache)

/-- Run one token position through a single transformer layer.

    x      : [dModel]  current hidden state
    layer  : LayerWeights (pre-built analytically)
    cache  : per-head KV cache for this layer
    heads  : number of attention heads (runtime)
    headDim: dims per head                  (heads * headDim = dModel as invariant)

    Returns (new hidden state, updated cache). -/
def layerStep {dModel ffnHidden : Nat}
    (heads headDim : Nat)
    (x     : Vector dModel .float32)
    (layer : LayerWeights dModel ffnHidden)
    (cache : LayerCache)
    : TensorM (Vector dModel .float32 × LayerCache) := do
  -- ① Pre-LN + Q/K/V projections
  let xNorm ← layerNorm x
  let xLn   ← mul xNorm layer.lnAttn
  let qProj ← project xLn layer.wQ
  let kProj ← project xLn layer.wK
  let vProj ← project xLn layer.wV

  -- ② Bridge to plain arrays for variable-length attention
  let qArr := tensorToArr qProj
  let kArr := tensorToArr kProj
  let vArr := tensorToArr vProj

  -- ③ Per-head attention (hull KV cache)
  let mut outParts : Array (Array Float32) := #[]
  let mut newCache := cache
  for h in [:heads] do
    let s := h * headDim
    let qH := qArr.extract s (s + headDim)
    let kH := kArr.extract s (s + headDim)
    let vH := vArr.extract s (s + headDim)
    let hc  := newCache.getD h (default : HeadCache)
    let out := hc.query qH               -- attend to past
    let hc' := hc.add kH vH hc.size     -- insert current k/v
    newCache := newCache.set! h hc'
    outParts := outParts.push out

  -- ④ Concat head outputs → [dModel] and output-project
  let outArr := outParts.foldl (· ++ ·) #[]
  let outVec ← arrToVector dModel outArr
  let outProj ← project outVec layer.wO
  let x2 ← add x outProj               -- residual

  -- ⑤ Pre-LN + FFN
  let x2Norm ← layerNorm x2
  let x2Ln   ← mul x2Norm layer.lnFFN
  let x2Row  ← reshapeUnsafe x2Ln [1, dModel]
  let h1     ← matmul x2Row layer.ff1   -- [1, ffnHidden]
  let h1Act  ← relu h1
  let h2     ← matmul h1Act layer.ff2   -- [1, dModel]
  let ffnOut ← reshapeUnsafe h2 [dModel]
  let x3     ← add x2 ffnOut            -- residual

  pure (x3, newCache)

end LLMComputer.Attention
