;; RNNoise's int8 matrix-vector products (sparse_cgemv8x4 in xiph/rnnoise vec.h, without the scale) in
;; WebAssembly SIMD, embedded in rnnoise.js as bytes. Rebuild: wat2wasm gemv.wat -o gemv.wasm (WABT).
(module
  (memory (export "memory") 1)
  ;; gemv(w, idx, x, out, rows): int32 out[r] = Σ w[r, c] · x[c] over upstream's 8×4-blocked int8 layout
  ;; (sparse_cgemv8x4 without the scale): per 8-row block, idx holds a count, then the column of each 4-wide block.
  (func (export "gemv") (param $w i32) (param $idx i32) (param $x i32) (param $out i32) (param $rows i32)
    (local $i i32) (local $n i32) (local $x16 v128) (local $wv v128)
    (local $a v128) (local $b v128) (local $c v128) (local $d v128)
    (loop $block
      (local.set $n (i32.load (local.get $idx)))
      (local.set $idx (i32.add (local.get $idx) (i32.const 4)))
      (local.set $a (v128.const i32x4 0 0 0 0)) (local.set $b (v128.const i32x4 0 0 0 0))
      (local.set $c (v128.const i32x4 0 0 0 0)) (local.set $d (v128.const i32x4 0 0 0 0))
      (block $done
        (loop $cols
          (br_if $done (i32.eqz (local.get $n)))
          (local.set $x16 (i16x8.extend_low_i8x16_s (i32x4.splat (i32.load (i32.add (local.get $x) (i32.load (local.get $idx)))))))
          (local.set $wv (v128.load (local.get $w)))
          (local.set $a (i32x4.add (local.get $a) (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s (local.get $wv)) (local.get $x16))))
          (local.set $b (i32x4.add (local.get $b) (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $wv)) (local.get $x16))))
          (local.set $wv (v128.load offset=16 (local.get $w)))
          (local.set $c (i32x4.add (local.get $c) (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s (local.get $wv)) (local.get $x16))))
          (local.set $d (i32x4.add (local.get $d) (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $wv)) (local.get $x16))))
          (local.set $w (i32.add (local.get $w) (i32.const 32)))
          (local.set $idx (i32.add (local.get $idx) (i32.const 4)))
          (local.set $n (i32.sub (local.get $n) (i32.const 1)))
          (br $cols)))
      ;; lanes hold half-rows: row = lane pair sum
      (v128.store (local.get $out) (i32x4.add
        (i8x16.shuffle 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 (local.get $a) (local.get $b))
        (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31 (local.get $a) (local.get $b))))
      (v128.store offset=16 (local.get $out) (i32x4.add
        (i8x16.shuffle 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 (local.get $c) (local.get $d))
        (i8x16.shuffle 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31 (local.get $c) (local.get $d))))
      (local.set $out (i32.add (local.get $out) (i32.const 32)))
      (local.set $i (i32.add (local.get $i) (i32.const 8)))
      (br_if $block (i32.lt_u (local.get $i) (local.get $rows))))))
