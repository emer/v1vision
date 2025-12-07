// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

//go:generate core generate -add-types

import (
	"cogentcore.org/lab/gosl/slbool"
	"github.com/emer/v1vision/fffb"
	"github.com/emer/v1vision/nxx1"
)

//gosl:start
//gosl:import "github.com/emer/v1vision/fffb"
//gosl:import "github.com/emer/v1vision/nxx1"

// KWTA contains all the parameters needed for computing FFFB
// (feedforward & feedback) inhibition that results in roughly
// k-Winner-Take-All behavior.
type KWTA struct {

	// On is whether to run kWTA or not.
	On slbool.Bool

	// Iters is the maximum number of iterations to perform.
	Iters int32 `default:"10"`

	// Threshold on delta-activation (change in activation) for stopping
	// updating of activations. Not used on GPU implementation.
	DelActThr float32 `default:"0.005"`

	// Time constant for integrating activation
	ActTau float32 `default:"3"`

	// Layer-level feedforward & feedback inhibition, applied over entire set of values.
	Layer fffb.FFFB `display:"inline"`

	// Pool-level (feature groups) feedforward and feedback inhibition.
	// applied within inner-most dimensions inside outer 2 dimensions.
	Pool fffb.FFFB `display:"inline"`

	// XX1 are the Noisy X/X+1 rate code activation function parameters.
	XX1 nxx1.Params `display:"inline"`

	// GBar are maximal conductances levels for channels.
	Gbar Chans `display:"inline"`

	// Erev are reversal potentials for each channel.
	Erev Chans `display:"inline"`

	// Erev - Act.Thr for each channel -- used in computing GeThrFromG among others
	ErevSubThr Chans `display:"-"`

	// Act.Thr - Erev for each channel -- used in computing GeThrFromG among others
	ThrSubErev Chans `display:"-" json:"-" xml:"-"`

	ActDt float32 `display:"-"; json"-" xml"-" desc:"integration rate = 1/ tau"`

	pad, pad1, pad2 float32
}

func (kp *KWTA) Defaults() {
	kp.On.SetBool(true)
	kp.Iters = 10 // 10 is typically sufficient.
	kp.DelActThr = 0.005
	kp.Layer.Defaults()
	kp.Pool.Defaults()
	kp.Layer.On.SetBool(true)
	kp.Layer.Gi = 1.5 // from lvis
	kp.Pool.On.SetBool(true)
	kp.Pool.Gi = 2.0
	kp.XX1.Defaults()
	kp.XX1.Gain = 80   // from lvis
	kp.XX1.NVar = 0.01 // from lvis
	kp.ActTau = 3
	kp.Gbar.SetAll(0.5, 0.1, 1.0, 1.0) // 0.5 is key for 1.0 inputs
	kp.Erev.SetAll(1.0, 0.3, 0.3, 0.1)
	kp.Update()
}

// Update must be called after any changes to parameters
func (kp *KWTA) Update() {
	kp.Layer.Update()
	kp.Pool.Update()
	kp.XX1.Update()
	kp.ErevSubThr.SetFromOtherMinus(kp.Erev, kp.XX1.Thr)
	kp.ThrSubErev.SetFromMinusOther(kp.XX1.Thr, kp.Erev)
	kp.ActDt = 1 / kp.ActTau
}

// GeThrFromG computes the threshold for Ge based on other conductances
func (kp *KWTA) GeThrFromG(gi float32) float32 {
	ge := ((kp.Gbar.I*gi*kp.ErevSubThr.I + kp.Gbar.L*kp.ErevSubThr.L) / kp.ThrSubErev.E)
	return ge
}

// ActFromG computes rate-coded activation Act from conductances Ge and Gi
func (kp *KWTA) ActFromG(geThr, ge, act float32, delAct *float32) float32 {
	nwAct := kp.XX1.NoisyXX1(ge*kp.Gbar.E - geThr)
	*delAct = kp.ActDt * (nwAct - act)
	nwAct = act + *delAct
	return nwAct
}

//gosl:end
