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

// Directions are the motion directions, in feature order,
// as represented in the Star and FullField outputs.
type Directions int32 //enums:enum

const (
	Left Directions = iota
	Right
	Down
	Up
)

// Params has the motion parameters for retinal starburst amacrine
// cells (SAC) that compute centrifugal motion flow from each point.
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

	// NormInteg is the integrated normalization value -- updated in FullFieldInteg
	NormInteg float32 `edit:"-"`

	// DoGSumScalarIndex is the index into the V1Vision Scalars output for
	// Sum of DoG activity, used for normalizing.
	DoGSumScalarIndex int `edit:"-"`

	// FFScalarIndex is the index into the V1Vision Scalars output for FullField
	FFScalarIndex int `edit:"-"`
}

func (pr *Params) Defaults() {
	// note: these values have been optimized on axon deepspace model:
	pr.SlowTau = 4
	pr.FastTau = 2
	pr.Gain = 20
	pr.FullGain = 1
	pr.IntegTau = 6
}

// FullFieldInteg computes a full-field integration of instantaneous
// MotionFullField results, in scalars input at FFScalarIndex
// Resulting integ tensor is 4 values (2x2) with left, right, bottom, top units.
// integ = integrated full-field values over time
// visNormInteg = integrated visNorm, actually used for normalization
func (pr *Params) FullFieldInteg(ndata int, scalars, integ *tensor.Float32) {
	idt := 1.0 / pr.IntegTau
	integ.SetShapeSizes(ndata, 2, 2)
	for di := range ndata {
		visNorm := scalars.Value(pr.DoGSumScalarIndex, di)
		if pr.NormInteg == 0 {
			pr.NormInteg = visNorm
		} else {
			pr.NormInteg += idt * (visNorm - pr.NormInteg)
		}
		vnf := pr.FullGain
		if pr.NormInteg > 0 {
			vnf /= pr.NormInteg
		}

		act := func(v float32) float32 { return vnf * v }
		l := scalars.Value(pr.FFScalarIndex+0, di)
		r := scalars.Value(pr.FFScalarIndex+1, di)
		if l > r {
			l = act(l - r)
			r = 0
		} else {
			r = act(r - l)
			l = 0
		}
		b := scalars.Value(pr.FFScalarIndex+2, di)
		u := scalars.Value(pr.FFScalarIndex+3, di)
		if b > u {
			b = act(b - u)
			u = 0
		} else {
			u = act(u - b)
			b = 0
		}

		integf := func(y, x int, v float32) {
			vi := integ.Value(di, y, x)
			vi += idt * (v - vi)
			integ.Set(vi, di, y, x)
		}
		integf(0, 0, l)
		integf(0, 1, r)
		integf(1, 0, b)
		integf(1, 1, u)
	}
}
