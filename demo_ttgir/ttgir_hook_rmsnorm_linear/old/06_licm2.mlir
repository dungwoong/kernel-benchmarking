#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>



#loc = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":20:1)
#loc1 = loc(unknown)
#loc25 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":65:29)
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>

#linear_guess = #ttg.linear<{register = [[8,0],[0,8],[0,16],[0,32],[0,1]], lane = [[0,2],[0,4],[1,0],[2,0],[4,0]], warp = [[16,0],[32,0],[64,0]], block = []}>
#linear_guess_3d = #ttg.linear<{register = [[8,0,0],[0,4,0],[0,8,0],[0,16,0],[0,0,1]], lane = [[0,1,0],[0,2,0],[1,0,0],[2,0,0],[4,0,0]], warp = [[16,0,0],[32,0,0],[64,0,0]], block = []}>
#linear_guess_2d = #ttg.linear<{register = [[8,0],[0,4],[0,8],[0,16]], lane = [[0,1],[0,2],[1,0],[2,0],[4,0]], warp = [[16,0],[32,0],[64,0]], block = []}>
#linear_guess_split = #ttg.linear<{register = [[8,0,0],[0,1,0],[0,2,0],[0,4,0]], lane = [[0,0,1],[0,0,2],[1,0,0],[2,0,0],[4,0,0]], warp = [[16,0,0],[32,0,0],[64,0,0]], block = []}>

#linear_guess_r = #ttg.linear<{
  register = [[8,0,0,0], [0,1,0,0], [0,2,0,0], [0,4,0,0], [0,0,0,1]],
  lane     = [[0,0,1,0], [0,0,2,0], [1,0,0,0], [2,0,0,0], [4,0,0,0]],
  warp     = [[16,0,0,0], [32,0,0,0], [64,0,0,0]],
  block    = []
}>

#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#loc36 = loc("x"(#loc))
#loc37 = loc("y"(#loc))
#loc38 = loc("out"(#loc))
#loc39 = loc("eps"(#loc))
#loc58 = loc("sum_1"(#loc25))
#loc73 = loc(callsite(#loc1 at #loc58))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_helion_rmsnorm_lin_kernel(%x: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("x"(#loc)), %y: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("y"(#loc)), %out: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("out"(#loc)), %eps: f32 loc("eps"(#loc))) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64 loc(#loc1)
    %c4096_i64 = arith.constant 4096 : i64 loc(#loc1)
    %c4096_i32 = arith.constant 4096 : i32 loc(#loc1)
    %end_pid = arith.constant 512 : i32 loc(#loc67)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %block_size = arith.constant 4 : i32 loc(#loc68)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc5)
    %cst = arith.constant dense<2.44140625E-4> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #dot0}>> loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x4xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>> loc(#loc1)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma> loc(#loc1)
    %x_desc = tt.make_tensor_descriptor %x, [%c4096_i32, %c4096_i32], [%c4096_i64, %c1_i64] : <bf16>, <128x64xbf16> loc(#loc42)
    %y_desc = tt.make_tensor_descriptor %y, [%c4096_i32, %c4096_i32], [%c4096_i64, %c1_i64] : <bf16>, <256x64xbf16> loc(#loc43)
    %out_desc = tt.make_tensor_descriptor %out, [%c4096_i32, %c4096_i32], [%c4096_i64, %c1_i64] : <bf16>, <128x256xbf16> loc(#loc44)
    %start_pid = tt.get_program_id x : i32 loc(#loc45)
    %start_pid_2 = arith.muli %start_pid, %block_size : i32 loc(#loc45)
    %end_pid_3 = arith.addi %start_pid_2, %block_size : i32 loc(#loc69)
    %0 = arith.cmpi sgt, %end_pid_3, %end_pid : i32 loc(#loc11)
    %1 = arith.select %0, %end_pid, %end_pid_3 : i32 loc(#loc12)
    scf.for %virtual_pid = %start_pid_2 to %1 step %c1_i32  : i32 {
      %pid_0 = arith.remsi %virtual_pid, %c32_i32 : i32 loc(#loc47)
      %pid_1 = arith.divsi %virtual_pid, %c32_i32 : i32 loc(#loc48)
      %offset_0 = arith.muli %pid_0, %c128_i32 : i32 loc(#loc49)
      %offset_1 = arith.muli %pid_1, %c256_i32 : i32 loc(#loc50)
      %sum_acc:2 = scf.for %offset_2 = %c0_i32 to %c4096_i32 step %c64_i32 iter_args(%acc = %cst_1, %sum_acc_7 = %cst_0) -> (tensor<128x256xf32, #mma>, tensor<128x4xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>>)  : i32 {
        %xTile = tt.descriptor_load %x_desc[%offset_0, %offset_2] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #blocked> loc(#loc52)
        
        // want to try making an equivalent linear layout
        %xTile_8 = ttg.convert_layout %xTile : tensor<128x64xbf16, #blocked> -> tensor<128x64xbf16, #dot0> loc(#loc52)
        %xTile_8_lin = ttg.convert_layout %xTile_8 : tensor<128x64xbf16, #dot0> -> tensor<128x64xbf16, #linear_guess>
        %xTile_r = tt.reshape %xTile_8_lin : tensor<128x64xbf16, #linear_guess> -> tensor<128x8x4x2xbf16, #linear_guess_r>

        %load_1 = tt.descriptor_load %y_desc[%offset_1, %offset_2] : !tt.tensordesc<256x64xbf16> -> tensor<256x64xbf16, #blocked> loc(#loc53)
        %load_1_9 = tt.trans %load_1 {order = array<i32: 1, 0>} : tensor<256x64xbf16, #blocked> -> tensor<64x256xbf16, #blocked2> loc(#loc54)
        %load_1_10 = ttg.local_alloc %load_1_9 : (tensor<64x256xbf16, #blocked2>) -> !ttg.memdesc<64x256xbf16, #shared1, #smem> loc(#loc54)
        // %acc_11 = ttg.convert_layout %acc : tensor<128x256xf32, #blocked1> -> tensor<128x256xf32, #mma> loc(#loc71)
        %acc_12 = ttng.warp_group_dot %xTile_8, %load_1_10, %acc {inputPrecision = 0 : i32} : tensor<128x64xbf16, #dot0> * !ttg.memdesc<64x256xbf16, #shared1, #smem> -> tensor<128x256xf32, #mma> loc(#loc56)
        // %acc_13 = ttg.convert_layout %acc_12 : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked1> loc(#loc56)
        // %v_0 = arith.mulf %xTile_8_lin, %xTile_8_lin : tensor<128x64xbf16, #linear_guess> loc(#loc57)
        %v_0 = arith.mulf %xTile_r, %xTile_r : tensor<128x8x4x2xbf16, #linear_guess_r>

        // STAGE 1: THREAD SUM OVER 16 ELEMENTS
        %red1 = "tt.reduce"(%v_0) ({
          ^bb0(%a: bf16, %b: bf16):
            %s = arith.addf %a, %b : bf16
            tt.reduce.return %s : bf16
        }) {axis = 1 : i32} : (tensor<128x8x4x2xbf16, #linear_guess_r>) -> tensor<128x4x2xbf16, #ttg.slice<{dim = 1, parent = #linear_guess_r}>>
        
        %red2 = "tt.reduce"(%red1) ({
          ^bb0(%a: bf16, %b: bf16):
            %s = arith.addf %a, %b : bf16
            tt.reduce.return %s : bf16
        }) {axis = 2 : i32} : (tensor<128x4x2xbf16, #ttg.slice<{dim = 1, parent = #linear_guess_r}>>)
          -> tensor<128x4xbf16, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>>

        %v_1 = arith.extf %red2 : tensor<128x4xbf16, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>> to tensor<128x4xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>> loc(#loc59)
        %sum_acc_14 = arith.addf %sum_acc_7, %v_1 : tensor<128x4xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>> loc(#loc60)
        scf.yield %acc_12, %sum_acc_14 : tensor<128x256xf32, #mma>, tensor<128x4xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>> loc(#loc17)
      } {tt.flatten} loc(#loc70)

      // WARP REDUCE
      // STAGE 2: WARP, results in nested ttg.slice
      %sum_1_raw = "tt.reduce"(%sum_acc#1) <{axis = 1 : i32}> ({
        ^bb0(%a: f32, %b: f32):
        %s = arith.addf %a, %b : f32
        tt.reduce.return %s : f32
      }) : (tensor<128x4xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>}>>
      %sum_1 = ttg.convert_layout %sum_1_raw : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 1, parent = #linear_guess_r}>}>}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #dot0}>>

      %v_4 = arith.mulf %sum_1, %cst : tensor<128xf32, #ttg.slice<{dim = 1, parent = #dot0}>> loc(#loc61)
      %v_5 = tt.splat %eps : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #dot0}>> loc(#loc62)
      %v_5_4 = arith.addf %v_4, %v_5 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #dot0}>> loc(#loc62)
      %v_6 = tt.extern_elementwise %v_5_4 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #dot0}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #dot0}>> loc(#loc63)
      %subscript = tt.expand_dims %v_6 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<128x1xf32, #dot0> loc(#loc64)
      %v_7 = ttg.convert_layout %subscript : tensor<128x1xf32, #dot0> -> tensor<128x1xf32, #mma> loc(#loc65)
      %v_7_5 = tt.broadcast %v_7 : tensor<128x1xf32, #mma> -> tensor<128x256xf32, #mma> loc(#loc65)
      %v_7_6 = arith.mulf %sum_acc#0, %v_7_5 : tensor<128x256xf32, #mma> loc(#loc65)
      %v_8 = arith.truncf %v_7_6 : tensor<128x256xf32, #mma> to tensor<128x256xbf16, #mma> loc(#loc66)
      %2 = ttg.convert_layout %v_8 : tensor<128x256xbf16, #mma> -> tensor<128x256xbf16, #blocked3> loc(#loc35)
      tt.descriptor_store %out_desc[%offset_0, %offset_1], %2 : !tt.tensordesc<128x256xbf16>, tensor<128x256xbf16, #blocked3> loc(#loc35)
    } loc(#loc5)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":31:5)
#loc3 = loc("/opt/conda/lib/python3.11/site-packages/triton/language/standard.py":43:12)
#loc4 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":32:18)
#loc5 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":37:5)
#loc6 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":22:14)
#loc7 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":24:14)
#loc8 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":26:16)
#loc9 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":33:17)
#loc10 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":34:15)
#loc11 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":35:8)
#loc12 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":35:5)
#loc13 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":40:17)
#loc14 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":41:17)
#loc15 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":42:20)
#loc16 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":43:20)
#loc17 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":52:9)
#loc18 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":58:21)
#loc19 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":60:33)
#loc20 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":60:22)
#loc21 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":45:15)
#loc22 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":61:19)
#loc23 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":63:19)
#loc24 = loc("/opt/conda/lib/python3.11/site-packages/triton/language/standard.py":293:12)
#loc26 = loc("/opt/conda/lib/python3.11/site-packages/triton/language/standard.py":263:12)
#loc27 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":66:19)
#loc28 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":67:23)
#loc29 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":70:15)
#loc30 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":71:15)
#loc31 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":72:15)
#loc32 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":74:21)
#loc33 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":75:15)
#loc34 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":77:15)
#loc35 = loc("/tmp/torchinductor_wangke61/gk/cgkjnt7qu4ajuarebji3vtpb6h3yyov7p3voyuwccegtxvj3h2yf.py":78:9)
#loc40 = loc("total_pids"(#loc2))
#loc41 = loc("block_size"(#loc4))
#loc42 = loc("x_desc"(#loc6))
#loc43 = loc("y_desc"(#loc7))
#loc44 = loc("out_desc"(#loc8))
#loc45 = loc("start_pid"(#loc9))
#loc46 = loc("end_pid"(#loc10))
#loc47 = loc("pid_0"(#loc13))
#loc48 = loc("pid_1"(#loc14))
#loc49 = loc("offset_0"(#loc15))
#loc50 = loc("offset_1"(#loc16))
#loc51 = loc("acc"(#loc17))
#loc52 = loc("xTile"(#loc18))
#loc53 = loc("load_1"(#loc19))
#loc54 = loc("load_1"(#loc20))
#loc55 = loc("acc"(#loc21))
#loc56 = loc("acc"(#loc22))
#loc57 = loc("v_0"(#loc23))
#loc59 = loc("v_1"(#loc27))
#loc60 = loc("sum_acc"(#loc28))
#loc61 = loc("v_4"(#loc29))
#loc62 = loc("v_5"(#loc30))
#loc63 = loc("v_6"(#loc31))
#loc64 = loc("subscript"(#loc32))
#loc65 = loc("v_7"(#loc33))
#loc66 = loc("v_8"(#loc34))
#loc67 = loc("end_pid"(#loc40))
#loc68 = loc(callsite(#loc3 at #loc41))
#loc69 = loc("end_pid"(#loc46))
#loc70 = loc("sum_acc"(#loc51))
#loc71 = loc("acc"(#loc55))
#loc72 = loc(callsite(#loc24 at #loc58))
#loc74 = loc(callsite(#loc26 at #loc72))
