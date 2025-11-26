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

//go:generate core generate -add-types -gosl

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

	// FullGain is multiplier for FullField
	FullGain float32

	// IntegTau is the integration time constant for integrating
	// the normalization and full field values over frames, to get
	// a more consistent value.
	IntegTau float32
}

func (pr *Params) Defaults() {
	// note: these values have been optimized on axon deepspace model:
	pr.SlowTau = 4
	pr.FastTau = 2
	pr.Gain = 20
	pr.FullGain = 1
	pr.IntegTau = 6
}

// IntegrateFrame integrates one frame of values into fast and slow tensors.
// returns the raw visual energy from the input, which is used for normalization
// of the full field results.
func (pr *Params) IntegrateFrame(slow, fast, in *tensor.Float32) float32 {
	fdt := 1.0 / pr.FastTau
	sdt := 1.0 / pr.SlowTau
	tensor.SetShapeFrom(slow, in)
	tensor.SetShapeFrom(fast, in)
	insum := float32(0)
	n := in.Len()
	for i := range n {
		v := in.Value1D(i)
		insum += v
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
	return insum
}

// StarMotion computes starburst-style motion for given slow and fast
// integrated values, which must be 3D tensors: filter, X, Y
// The resulting output is stored in 2x2 inner dimensions of output
// with left, right, down, up motion signals.
func (pr *Params) StarMotion(out, slow, fast *tensor.Float32) {
	nf := slow.DimSize(0)
	ny := slow.DimSize(1)
	nx := slow.DimSize(2)
	out.SetShapeSizes(nf, ny-1, nx-1, 2, 2)
	for flt := range nf {
		for y := range ny - 1 {
			for x := range nx - 1 {
				sl := slow.Value(flt, y, x)
				sr := slow.Value(flt, y, x+1)
				fl := fast.Value(flt, y, x)
				fr := fast.Value(flt, y, x+1)
				minact := min(min(min(sl, fl), sr), fr)
				ld := fl - sl
				rd := fr - sr
				if ld > rd {
					v := minact * pr.Gain * (ld - rd)
					out.Set(v, flt, y, x, 0, 0)
					out.Set(0, flt, y, x, 0, 1)
				} else {
					v := minact * pr.Gain * (rd - ld)
					out.Set(v, flt, y, x, 0, 1)
					out.Set(0, flt, y, x, 0, 0)
				}
			}
		}
		for x := range nx - 1 {
			for y := range ny - 1 {
				sb := slow.Value(flt, y, x)
				st := slow.Value(flt, y+1, x)
				fb := fast.Value(flt, y, x)
				ft := fast.Value(flt, y+1, x)
				minact := min(min(min(sb, fb), st), ft)
				bd := fb - sb
				td := ft - st
				if bd > td {
					v := minact * pr.Gain * (bd - td)
					out.Set(v, flt, y, x, 0, 2)
					out.Set(0, flt, y, x, 0, 3)
				} else {
					v := minact * pr.Gain * (td - bd)
					out.Set(v, flt, y, x, 0, 3)
					out.Set(0, flt, y, x, 0, 2)
				}
			}
		}
	}
}

// FullField computes a full-field summary of output from StarMotion.
// Result is just a 2x2 output with left, right, bottom, top units.
// Opposite directions compete.
// insta = instantaneous full-field values per this frame
// integ = integrated full-field values over time
// visNorm = total filter energy normalization factor, e.g., from FilterEnergy
// visNormInteg = integrated visNorm, actually used for normalization
func (pr *Params) FullField(insta, integ, star *tensor.Float32, visNorm float32, visNormInteg *float32) {
	insta.SetShapeSizes(2, 2)
	integ.SetShapeSizes(2, 2)
	nf := star.DimSize(0)
	ny := star.DimSize(1)
	nx := star.DimSize(2)
	tensor.SetAllFloat64(insta, 0)

	idt := 1.0 / pr.IntegTau
	if *visNormInteg == 0 {
		*visNormInteg = visNorm
	} else {
		*visNormInteg += idt * (visNorm - *visNormInteg)
	}
	vnf := pr.FullGain
	if *visNormInteg > 0 {
		vnf /= *visNormInteg
	}

	for flt := range nf {
		for y := range ny {
			for x := range nx {
				l := star.Value(flt, y, x, 0, 0)
				r := star.Value(flt, y, x, 0, 1)
				if l > r {
					insta.SetAdd(l-r, 0, 0)
				} else {
					insta.SetAdd(r-l, 0, 1)
				}
				b := star.Value(flt, y, x, 1, 0)
				t := star.Value(flt, y, x, 1, 1)
				if b > t {
					insta.SetAdd(b-t, 1, 0)
				} else {
					insta.SetAdd(t-b, 1, 1)
				}
			}
		}
	}
	act := func(v float32) float32 { return vnf * v }
	l := insta.Value(0, 0)
	r := insta.Value(0, 1)
	if l > r {
		insta.Set(act(l-r), 0, 0)
		insta.Set(0, 0, 1)
	} else {
		insta.Set(act(r-l), 0, 1)
		insta.Set(0, 0, 0)
	}
	b := insta.Value(1, 0)
	t := insta.Value(1, 1)
	if b > t {
		insta.Set(act(b-t), 1, 0)
		insta.Set(0, 1, 1)
	} else {
		insta.Set(act(t-b), 1, 1)
		insta.Set(0, 1, 0)
	}

	//////// integration
	for i := range 4 {
		v := insta.Value1D(i)
		vi := integ.Value1D(i)
		vi += idt * (v - vi)
		integ.Set1D(vi, i)
	}
}
