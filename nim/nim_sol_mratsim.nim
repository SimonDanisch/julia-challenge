# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

## This files gives basic tensor library functionality, because yes we can
import strformat, macros, sequtils, random

type
  Tensor[Rank: static[int], T] = object
    ## Tensor data structure stored on Cpu
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the tensor. Note: offset can be negative, in particular for slices.
    ##   - ``storage``: A data storage for the tensor
    ##   - Rank is part of the type for optimization purposes
    ##
    ## Warning ⚠:
    ##   Assignment ```var a = b``` does not copy the data. Data modification on one tensor will be reflected on the other.
    ##   However modification on metadata (shape, strides or offset) will not affect the other tensor.
    shape: array[Rank, int]
    strides: array[Rank, int]
    offset: int
    storage: CpuStorage[T]

  CpuStorage*{.shallow.}[T] = object
    ## Data storage for the tensor, copies are shallow by default
    data*: seq[T]

template tensor(result: var Tensor, shape: array) =
  result.shape = shape

  var accum = 1
  for i in countdown(Rank - 1, 0):
    result.strides[i] = accum
    accum *= shape[i]

func newTensor*[Rank: static[int], T](shape: array[Rank, int]): Tensor[Rank, T] =
  tensor(result, shape)
  result.storage.data = newSeq[T](shape.product)

proc rand[T: object|tuple](max: T): T =
  ## A generic random function for any stack object or tuple
  ## that initialize all fields randomly
  result = max
  for field in result.fields:
    field = rand(field)

proc randomTensor*[Rank: static[int], T](shape: array[Rank, int], max: T): Tensor[Rank, T] =
  tensor(result, shape)
  result.storage.data = newSeqWith(shape.product, T(rand(max)))

func getIndex[Rank, T](t: Tensor[Rank, T], idx: array[Rank, int]): int {.inline.} =
  ## Convert [i, j, k, l ...] to the memory location referred by the index
  result = t.offset
  for i in 0 ..< t.Rank:
    {.unroll.} # I'm sad this doesn't work yet
    result += t.strides[i] * idx[i]

func `[]`[Rank, T](t: Tensor[Rank, T], idx: array[Rank, int]): T {.inline.}=
  ## Index tensor
  t.storage.data[t.getIndex(idx)]

func `[]=`[Rank, T](t: var Tensor[Rank, T], idx: array[Rank, int], val: T) {.inline.}=
  ## Index tensor
  t.storage.data[t.getIndex(idx)] = val

template `[]`[T: SomeNumber](x: T, idx: varargs[int]): T =
  ## "Index" scalars
  x

func shape(x: SomeNumber): array[1, int] = [1]

func bcShape[R1, R2: static[int]](x: array[R1, int]; y: array[R2, int]): auto =
  when R1 > R2:
    result = x
    for i, idx in result.mpairs:
      if idx == 1 and y[i] != 1:
        idx = y[i]
  else:
    result = y
    for i, idx in result.mpairs:
      if idx == 1 and x[i] != 1:
        idx = x[i]

macro getBroadcastShape(x: varargs[typed]): untyped =
  assert x.len >= 2
  result = nnkDotExpr.newTree(x[0], ident"shape")
  for i in 1 ..< x.len:
    let xi = x[i]
    result = quote do: bcShape(`result`, `xi`.shape)

func bc[R1, R2: static[int], T](t: Tensor[R1, T], shape: array[R2, int]): Tensor[R2, T] =
  ## Broadcast tensors
  result.shape = shape
  for i in 0 ..< R1:
    if t.shape[i] == 1 and shape[i] != 1:
      result.strides[i] = 0
    else:
      result.strides[i] = t.strides[i]
      if t.shape[i] != result.shape[i]:
        raise newException(ValueError, "The broadcasted size of the tensor must match existing size for non-singleton dimension")
  result.offset = t.offset
  result.storage = t.storage

func bc[Rank; T: SomeNumber](x: T, shape: array[Rank, int]): T {.inline.}=
  ## "Broadcast" scalars
  x

func product(x: varargs[int]): int =
  result = 1
  for val in x: result *= val

proc replaceNodes(ast: NimNode, values: NimNode, containers: NimNode): NimNode =
  # Args:
  #   - The full syntax tree
  #   - an array of replacement value
  #   - an array of identifiers to replace
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of {nnkIdent, nnkSym}:
      for i, c in containers:
        if node.eqIdent($c):
          return values[i]
      return node
    of nnkEmpty: return node
    of nnkLiterals: return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)

proc pop*(tree: var NimNode): NimNode =
  ## varargs[untyped] consumes all arguments so the actual value should be popped
  ## https://github.com/nim-lang/Nim/issues/5855
  result = tree[tree.len-1]
  tree.del(tree.len-1)

func nb_elems[N: static[int], T](x: typedesc[array[N, T]]): static[int] =
  N

macro broadcastImpl(output: untyped, inputs_body: varargs[untyped]): untyped =
  ## If output is empty node it will return a value
  ## otherwise, result will be assigned in-place to output
  let
    in_place = newLit output.kind != nnkEmpty

  var
    inputs = inputs_body
    body = inputs.pop()

  let
    shape = genSym(nskLet, "broadcast_shape__")
    coord = genSym(nskVar, "broadcast_coord__")

  var doBroadcast = newStmtList()
  var bcInputs = nnkArgList.newTree()
  for input in inputs:
    let broadcasted = genSym(nskLet, "broadcast_" & $input & "__")
    doBroadcast.add newLetStmt(
      broadcasted,
      newCall(ident"bc", input, shape)
    )
    bcInputs.add nnkBracketExpr.newTree(broadcasted, coord)

  body = body.replaceNodes(bcInputs, inputs)

  result = quote do:
    block:
      let `shape` = getBroadcastShape(`inputs`)
      const rank = `shape`.type.nb_elems
      var `coord`: array[rank, int] # Current coordinates in the n-dimensional space
      `doBroadcast`

      when not `in_place`:
        var output = newTensor[rank, type(`body`)](`shape`)
      else:
        assert `output`.shape == `shape`
      var counter = 0

      while counter < `shape`.product:
        # Assign for the current iteration
        when not `in_place`:
          output[`coord`] = `body`
        else:
          `output`[`coord`] = `body`

        # Compute the next position
        for k in countdown(rank - 1, 0):
          if `coord`[k] < `shape`[k] - 1:
            `coord`[k] += 1
            break
          else:
            `coord`[k] = 0
        inc counter

      # Now return the value
      when not `in_place`:
        output

macro broadcast(inputs_body: varargs[untyped]): untyped =
  getAST(broadcastImpl(newEmptyNode(), inputs_body))

macro materialize(output: var Tensor, inputs_body: varargs[untyped]): untyped =
  getAST(broadcastImpl(output, inputs_body))

#################################################################################

import math
proc sanityChecks() =
  # Sanity checks

  let x = randomTensor([1, 2, 3], 10)
  let y = randomTensor([5, 2], 10)

  echo x # (shape: [1, 2, 3], strides: [6, 3, 1], offset: 0, storage: (data: @[1, 10, 5, 5, 7, 3]))
  echo y # (shape: [5, 2], strides: [2, 1], offset: 0, storage: (data: @[8, 3, 7, 9, 3, 8, 5, 3, 7, 1]))

  block: # Simple assignation
    echo "\nSimple assignation"
    let a = broadcast(x, y):
      x * y

    echo a # (shape: [5, 2, 3], strides: [6, 3, 1], offset: 0, storage: (data: @[8, 80, 40, 15, 21, 9, 7, 70, 35, 45, 63, 27, 3, 30, 15, 40, 56, 24, 5, 50, 25, 15, 21, 9, 7, 70, 35, 5, 7, 3]))

  block: # In-place, similar to Julia impl
    echo "\nIn-place, similar to Julia impl"
    var a = newTensor[3, int]([5, 2, 3])
    materialize(a, x, y):
      x * y

    echo a

  block: # Complex multi statement with type conversion
    echo "\nComplex multi statement with type conversion"
    let a = broadcast(x, y):
      let c = cos x.float64
      let s = sin y.float64

      sqrt(c.pow(2) + s.pow(2))

    echo a # (shape: [5, 2, 3], strides: [6, 3, 1], offset: 0, storage: (data: @[1.12727828058919, 1.297255090978019, 1.029220081237957, 0.3168265963213802, 0.7669963922853442, 0.9999999999999999, 0.8506221091780486, 1.065679324094626, 0.7156085706291233, 0.5003057878335346, 0.859191628789455, 1.072346394223034, 0.5584276483137685, 0.8508559734652587, 0.3168265963213802, 1.029220081237957, 1.243864280886628, 1.399612404734566, 1.100664502137075, 1.274196529364651, 1.0, 0.3168265963213802, 0.7669963922853442, 0.9999999999999999, 0.8506221091780486, 1.065679324094626, 0.7156085706291233, 0.8879964266455946, 1.129797339073468, 1.299291561428286]))

  block: # Variadic number of types with proc declaration inside
    echo "\nVariadic number of types with proc declaration inside"
    var u, v, w, x, y, z = randomTensor([3, 3], 10)

    let c = 2

    let a = broadcast(u, v, w, x, y, z):
      # ((u * v * w) div c) mod (if not zero (x - y + z) else 42)

      proc ifNotZero(val, default: int): int =
        if val == 0: default
        else: val

      let uvw_divc = u * v * w div c
      let xmypz = x - y + z

      uvw_divc mod ifNotZero(xmypz, 42)

    echo a # (shape: [3, 3], strides: [3, 1], offset: 0, storage: (data: @[0, 0, 0, 7, 4, 0, 0, 2, 0]))

#################################################################################

import math, random, times, stats, strformat
proc mainBench(nb_samples: int) =
  ## Bench with standard lib
  block: # Warmup - make sure cpu is on max perf
    let start = cpuTime()
    var foo = 123
    for i in 0 ..< 100_000_000:
      foo += i*i mod 456
      foo = foo mod 789

    # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
    let stop = cpuTime()
    echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)"

  let
    a = randomTensor([1000, 1000], 1.0)
    b = randomTensor([1000], 1.0)
    c = 1.0
  var output = newTensor[2, float64](a.shape)

  block: # Actual bench
    var stats: RunningStat
    for _ in 0 ..< nb_samples:
      let start = cpuTime()
      materialize(output, a, b, c):
        a + b - sin c
      let stop = cpuTime()
      stats.push stop - start

    echo &"\nTensors of Float64 bench"
    echo &"Collected {stats.n} samples"
    echo &"Average broadcast time: {stats.mean * 1000 :>4.3f}ms"
    echo &"Stddev  broadcast time: {stats.standardDeviationS * 1000 :>4.3f}ms"
    echo &"Min     broadcast time: {stats.min * 1000 :>4.3f}ms"
    echo &"Max     broadcast time: {stats.max * 1000 :>4.3f}ms"
    echo "\nDisplay output[[0,0]] to make sure it's not optimized away"
    echo output[[0, 0]]

proc geometryBench(nb_samples: int) =
  type Point3 = object
    x, y, z: float32

  template liftBinary(op: untyped): untyped =
    func `op`(a, b: Point3): Point3 {.inline.}=
      result.x = `op`(a.x, b.x)
      result.y = `op`(a.y, b.y)
      result.z = `op`(a.z, b.z)
    func `op`(a: Point3, b: float32): Point3 {.inline.}=
      result.x = `op`(a.x, b)
      result.y = `op`(a.y, b)
      result.z = `op`(a.z, b)
  template liftReduce(opName, op: untyped): untyped =
    func `opName`(a: Point3): float32 {.inline.}=
      a.x.`op`(a.y).`op`(a.z)

  liftBinary(`+`)
  liftBinary(`*`)
  liftBinary(`-`)
  liftReduce(sum, `+`)

  let
    a = randomTensor([1_000_000], Point3(x: 100, y: 100, z: 100))
    b = randomTensor([1_000_000], Point3(x: 100, y: 100, z: 100))
    c = 1.0'f32 # Julia has Point3 has float32 but C has float64
  var output = newTensor[1, float32](a.shape)

  block: # Custom function sqrt(sum(a .* b))
    func super_custom_func(a, b: Point3): float32 = sqrt sum(a * b)

    var stats: RunningStat
    for _ in 0 ..< nb_samples:
      let start = cpuTime()
      materialize(output, a, b):
        super_custom_func(a, b)
      let stop = cpuTime()
      stats.push stop - start

    echo &"\nTensor of 3D float32 points bench"
    echo &"Collected {stats.n} samples"
    echo &"Average broadcast time: {stats.mean * 1000 :>4.3f}ms"
    echo &"Stddev  broadcast time: {stats.standardDeviationS * 1000 :>4.3f}ms"
    echo &"Min     broadcast time: {stats.min * 1000 :>4.3f}ms"
    echo &"Max     broadcast time: {stats.max * 1000 :>4.3f}ms"
    echo "\nDisplay output[[0]] to make sure it's not optimized away"
    echo output[[0]]

when isMainModule:
  sanityChecks()
  echo "\n###################"
  echo "Benchmark"
  # {.passC: "-march=native" .} # uncomment to enable full optim (AVX/AVX2, ...)
  # randomize(seed = 0)
  mainBench(1_000)
  geometryBench(1_000)

  # Compile with
  # nim c -d:release nim/nim_sol_mratsim.nim     # for binary only
  # nim c -r -d:release nim/nim_sol_mratsim.nim  # for binary + running
