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

	// Gain is multiplier on the opponent difference for Star computation.
	Gain float32

	// FullGain is multiplier for FullField, which goes into a x/x+1 function.
	FullGain float32
}

func (pr *Params) Defaults() {
	pr.SlowTau = 20
	pr.FastTau = 10
	pr.Gain = 20
	pr.FullGain = 5
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

// FullField computes a full-field summary of output from StarMotion.
// Result is just a 2x2 output with left, right, bottom, top units.
// Opposite directions compete.
func (pr *Params) FullField(out, star *tensor.Float32) {
	out.SetShapeSizes(2, 2)
	ny := star.DimSize(0)
	nx := star.DimSize(1)
	tensor.SetAllFloat64(out, 0)
	for y := range ny {
		for x := range nx {
			l := star.Value(y, x, 0, 0)
			r := star.Value(y, x, 0, 1)
			if l > r {
				out.SetAdd(l-r, 0, 0)
			} else {
				out.SetAdd(r-l, 0, 1)
			}
			b := star.Value(y, x, 1, 0)
			t := star.Value(y, x, 1, 1)
			if b > t {
				out.SetAdd(b-t, 1, 0)
			} else {
				out.SetAdd(t-b, 1, 1)
			}
		}
	}
	act := func(v float32) float32 {
		v *= pr.FullGain
		return v / (v + 1)
	}
	l := out.Value(0, 0)
	r := out.Value(0, 1)
	if l > r {
		out.Set(act(l-r), 0, 0)
		out.Set(0, 0, 1)
	} else {
		out.Set(act(r-l), 0, 1)
		out.Set(0, 0, 0)
	}
	b := out.Value(1, 0)
	t := out.Value(1, 1)
	if b > t {
		out.Set(act(b-t), 1, 0)
		out.Set(0, 1, 1)
	} else {
		out.Set(act(t-b), 1, 1)
		out.Set(0, 1, 0)
	}
}
