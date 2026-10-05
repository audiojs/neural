;; The music guard's float32 products (guard.js: the classifier's upper convolutions through im2col, and its dense
;; layers) in WebAssembly SIMD, embedded in guard.js as bytes. Rebuild: wat2wasm gemm.wat -o gemm.wasm (WABT).
(module
  (memory (export "memory") 1)
  ;; gemm(w, x, b, y, rows, k, n): y[p, r] = max(0, b[r] + Σc w[r, c] · x[p, c]) for p < n, r < rows; w row-major
  ;; rows × k, x n × k, y n × rows; rows and k multiples of 4. Four rows and two positions at a time, lane l of a row
  ;; summing the columns c ≡ l (mod 4), the lanes then added (l0 + l1) + (l2 + l3), then the bias: guard.js's JS loop
  ;; does the same float ops in the same order
  (func (export "gemm") (param $w i32) (param $x i32) (param $b i32) (param $y i32) (param $rows i32) (param $k i32) (param $n i32)
    (local $r i32) (local $c i32) (local $p i32) (local $x0 i32) (local $x1 i32) (local $w0 i32) (local $kb i32)
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (local $b0 v128) (local $b1 v128) (local $b2 v128) (local $b3 v128)
    (local $u v128) (local $v v128) (local $s v128) (local $t v128)
    (local.set $kb (i32.shl (local.get $k) (i32.const 2)))
    (loop $pairs
      (local.set $x0 (i32.add (local.get $x) (i32.mul (local.get $p) (local.get $kb))))
      ;; the second of the pair: itself again when n is odd (computed, not stored)
      (local.set $x1 (select (i32.add (local.get $x0) (local.get $kb)) (local.get $x0) (i32.lt_u (i32.add (local.get $p) (i32.const 1)) (local.get $n))))
      (local.set $r (i32.const 0))
      (loop $rowblk
        (local.set $a0 (v128.const i32x4 0 0 0 0)) (local.set $a1 (v128.const i32x4 0 0 0 0))
        (local.set $a2 (v128.const i32x4 0 0 0 0)) (local.set $a3 (v128.const i32x4 0 0 0 0))
        (local.set $b0 (v128.const i32x4 0 0 0 0)) (local.set $b1 (v128.const i32x4 0 0 0 0))
        (local.set $b2 (v128.const i32x4 0 0 0 0)) (local.set $b3 (v128.const i32x4 0 0 0 0))
        (local.set $w0 (i32.add (local.get $w) (i32.mul (local.get $r) (local.get $kb))))
        (local.set $c (i32.const 0))
        (loop $cols
          (local.set $u (v128.load (i32.add (local.get $x0) (local.get $c))))
          (local.set $v (v128.load (i32.add (local.get $x1) (local.get $c))))
          (local.set $s (v128.load (i32.add (local.get $w0) (local.get $c))))
          (local.set $a0 (f32x4.add (local.get $a0) (f32x4.mul (local.get $s) (local.get $u))))
          (local.set $b0 (f32x4.add (local.get $b0) (f32x4.mul (local.get $s) (local.get $v))))
          (local.set $s (v128.load (i32.add (i32.add (local.get $w0) (local.get $kb)) (local.get $c))))
          (local.set $a1 (f32x4.add (local.get $a1) (f32x4.mul (local.get $s) (local.get $u))))
          (local.set $b1 (f32x4.add (local.get $b1) (f32x4.mul (local.get $s) (local.get $v))))
          (local.set $s (v128.load (i32.add (i32.add (local.get $w0) (i32.shl (local.get $kb) (i32.const 1))) (local.get $c))))
          (local.set $a2 (f32x4.add (local.get $a2) (f32x4.mul (local.get $s) (local.get $u))))
          (local.set $b2 (f32x4.add (local.get $b2) (f32x4.mul (local.get $s) (local.get $v))))
          (local.set $s (v128.load (i32.add (i32.add (local.get $w0) (i32.mul (local.get $kb) (i32.const 3))) (local.get $c))))
          (local.set $a3 (f32x4.add (local.get $a3) (f32x4.mul (local.get $s) (local.get $u))))
          (local.set $b3 (f32x4.add (local.get $b3) (f32x4.mul (local.get $s) (local.get $v))))
          (local.set $c (i32.add (local.get $c) (i32.const 16)))
          (br_if $cols (i32.lt_u (local.get $c) (local.get $kb))))
        ;; transpose-add the four row sums: lane q = Σ a_q
        (local.set $t (v128.load (i32.add (local.get $b) (i32.shl (local.get $r) (i32.const 2)))))
        (local.set $s (f32x4.add
          (f32x4.add (i8x16.shuffle 0 1 2 3 16 17 18 19 0 1 2 3 16 17 18 19 (local.get $a0) (local.get $a1)) (i8x16.shuffle 4 5 6 7 20 21 22 23 4 5 6 7 20 21 22 23 (local.get $a0) (local.get $a1)))
          (f32x4.add (i8x16.shuffle 8 9 10 11 24 25 26 27 8 9 10 11 24 25 26 27 (local.get $a0) (local.get $a1)) (i8x16.shuffle 12 13 14 15 28 29 30 31 12 13 14 15 28 29 30 31 (local.get $a0) (local.get $a1)))))
        (local.set $u (f32x4.add
          (f32x4.add (i8x16.shuffle 0 1 2 3 16 17 18 19 0 1 2 3 16 17 18 19 (local.get $a2) (local.get $a3)) (i8x16.shuffle 4 5 6 7 20 21 22 23 4 5 6 7 20 21 22 23 (local.get $a2) (local.get $a3)))
          (f32x4.add (i8x16.shuffle 8 9 10 11 24 25 26 27 8 9 10 11 24 25 26 27 (local.get $a2) (local.get $a3)) (i8x16.shuffle 12 13 14 15 28 29 30 31 12 13 14 15 28 29 30 31 (local.get $a2) (local.get $a3)))))
        (v128.store (i32.add (local.get $y) (i32.shl (i32.add (i32.mul (local.get $p) (local.get $rows)) (local.get $r)) (i32.const 2)))
          (f32x4.max (v128.const f32x4 0 0 0 0) (f32x4.add (local.get $t) (i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 (local.get $s) (local.get $u)))))
        (if (i32.lt_u (i32.add (local.get $p) (i32.const 1)) (local.get $n)) (then
          (local.set $s (f32x4.add
            (f32x4.add (i8x16.shuffle 0 1 2 3 16 17 18 19 0 1 2 3 16 17 18 19 (local.get $b0) (local.get $b1)) (i8x16.shuffle 4 5 6 7 20 21 22 23 4 5 6 7 20 21 22 23 (local.get $b0) (local.get $b1)))
            (f32x4.add (i8x16.shuffle 8 9 10 11 24 25 26 27 8 9 10 11 24 25 26 27 (local.get $b0) (local.get $b1)) (i8x16.shuffle 12 13 14 15 28 29 30 31 12 13 14 15 28 29 30 31 (local.get $b0) (local.get $b1)))))
          (local.set $u (f32x4.add
            (f32x4.add (i8x16.shuffle 0 1 2 3 16 17 18 19 0 1 2 3 16 17 18 19 (local.get $b2) (local.get $b3)) (i8x16.shuffle 4 5 6 7 20 21 22 23 4 5 6 7 20 21 22 23 (local.get $b2) (local.get $b3)))
            (f32x4.add (i8x16.shuffle 8 9 10 11 24 25 26 27 8 9 10 11 24 25 26 27 (local.get $b2) (local.get $b3)) (i8x16.shuffle 12 13 14 15 28 29 30 31 12 13 14 15 28 29 30 31 (local.get $b2) (local.get $b3)))))
          (v128.store (i32.add (local.get $y) (i32.shl (i32.add (i32.mul (i32.add (local.get $p) (i32.const 1)) (local.get $rows)) (local.get $r)) (i32.const 2)))
            (f32x4.max (v128.const f32x4 0 0 0 0) (f32x4.add (local.get $t) (i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 (local.get $s) (local.get $u)))))))
        (local.set $r (i32.add (local.get $r) (i32.const 4)))
        (br_if $rowblk (i32.lt_u (local.get $r) (local.get $rows))))
      (local.set $p (i32.add (local.get $p) (i32.const 2)))
      (br_if $pairs (i32.lt_u (local.get $p) (local.get $n))))))
