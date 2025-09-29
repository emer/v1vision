// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
package motion provides motion-filters based on retinal starburst amacrine
cells (SAC) that compute centrifugal motion flow from each point.
*/
package motion

import (
	"cogentcore.org/lab/tensor"
)

//go:generate core generate -add-types

// Params has the motion parameters.
type Params struct {

	// SlowTau is the time constant (in frames) for integrating
	// slow inhibitory inputs.
	SlowTau float32

	// FastTau is the time constant (in frames) for integrating
	// fast excitatory inputs.
	FastTau float32

	// Gain is multiplier on difference.
	Gain float32
}

func (pr *Params) Defaults() {
	pr.SlowTau = 10
	pr.FastTau = 5
	pr.Gain = 20
}

// IntegrateFrame integrates one frame of values into fast and slow tensors.
func (pr *Params) IntegrateFrame(slow, fast, in *tensor.Float32) {
	fdt := 1.0 / pr.FastTau
	sdt := 1.0 / pr.SlowTau
	tensor.SetShapeFrom(slow, in)
	tensor.SetShapeFrom(fast, in)
	n := in.Len()
	for i := range n {
		v := in.Value1D(i)
		s := slow.Value1D(i)
		f := fast.Value1D(i)
		if v > s {
			s = v
		} else {
			s += sdt * (v - s)
		}
		if v > f {
			f = v
		} else {
			f += fdt * (v - f)
		}
		slow.Set1D(s, i)
		fast.Set1D(f, i)
	}
}

// StarMotion computes starburst-style motion for given slow and fast
// integrated values, which must be 2D tensors.
// The resulting output is stored in 2x2 inner dimensions of output
// with left, right, down, up motion signals.
func (pr *Params) StarMotion(out, slow, fast *tensor.Float32) {
	ny := slow.DimSize(0)
	nx := slow.DimSize(1)
	out.SetShapeSizes(ny-1, nx-1, 2, 2)
	// fmt.Println(slow.ShapeSizes(), out.ShapeSizes())
	for y := range ny - 1 {
		for x := range nx - 1 {
			sl := slow.Value(y, x)
			sr := slow.Value(y, x+1)
			fl := fast.Value(y, x)
			fr := fast.Value(y, x+1)
			minact := min(min(min(sl, fl), sr), fr)
			ld := fl - sl
			rd := fr - sr
			if ld > rd {
				v := minact * min(1, pr.Gain*(ld-rd))
				out.Set(v, y, x, 0, 0)
				out.Set(0, y, x, 0, 1)
			} else {
				v := minact * min(1, pr.Gain*(rd-ld))
				out.Set(v, y, x, 0, 1)
				out.Set(0, y, x, 0, 0)
			}
		}
	}
	for x := range nx - 1 {
		for y := range ny - 1 {
			sb := slow.Value(y, x)
			st := slow.Value(y+1, x)
			fb := fast.Value(y, x)
			ft := fast.Value(y+1, x)
			minact := min(min(min(sb, fb), st), ft)
			bd := fb - sb
			td := ft - st
			if bd > td {
				v := minact * min(1, pr.Gain*(bd-td))
				out.Set(v, y, x, 0, 2)
				out.Set(0, y, x, 0, 3)
			} else {
				v := minact * min(1, pr.Gain*(td-bd))
				out.Set(v, y, x, 0, 3)
				out.Set(0, y, x, 0, 2)
			}
		}
	}
}
