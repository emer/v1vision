// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fffb

import "cogentcore.org/lab/gosl/slbool"

//go:generate core generate -add-types -gosl

//gosl:start

// FFFB parameterizes feedforward (FF) and feedback (FB) inhibition (FFFB)
// based on average (or maximum) netinput (FF) and activation (FB)
type FFFB struct {

	// enable this level of inhibition
	On slbool.Bool

	// Gi is the overall inhibition gain. This is main parameter to adjust to change
	// overall activation levels, as it scales both the the FF and FB factors uniformly.
	// 1.8 for layer, 2.0 for pool by default.
	Gi float32 `min:"0" default:"1.8,2"`

	// FF is the overall inhibitory contribution from feedforward inhibition.
	// This multiplies average netinput (i.e., synaptic drive into layer), which
	// anticipates upcoming changes in excitation. If set too high, it can make
	// activity slow to emerge. See also ff0 for a zero-point for this value.
	FF float32 `min:"0" default:"1"`

	// FB is the overall inhibitory contribution from feedback inhibition.
	// This multiplies average activation, thereby reacting to layer activation
	// levels and working like a thermostat (going up when the 'heat' in the layer
	// is too high).
	FB float32 `min:"0" default:"1"`

	// FBTau is the time constant in cycles (in milliseconds typically) for integrating
	// feedback inhibitory values, which prevents oscillations that otherwise occur.
	// The fast default of 1.4 should be used for most cases but sometimes a slower value
	// (3 or higher) can be more robust, especially when inhibition is strong or inputs
	// are more rapidly changing. (Tau is roughly 2/3 of the way to asymptote).
	FBTau float32 `min:"0" default:"1.4,3,5"`

	// MaxVsAvg determines the proportion of the maximum vs. average netinput to use in
	// the feedforward inhibition computation: 0 = all average, 1 = all max,
	// and values in between = proportional mix between average and max
	// (ff_netin = avg + ff_max_vs_avg * (max - avg)).
	// Including more max can be beneficial especially in situations where the average
	// can vary significantly but the activity should not -- max is more robust in many
	// situations but less flexible and sensitive to the overall distribution.
	// Max is better for cases more closely approximating single or strictly fixed
	// winner-take-all behavior: 0.5 is a good compromise in many cases and generally
	// requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0.
	MaxVsAvg float32 `default:"0,0.5,1"`

	// FF0 is the feedforward zero point for average netinput. Below this level, no FF
	// inhibition is computed based on avg netinput, and this value is subtraced from
	// the FF inhib contribution above this value. The 0.1 default should be good for most
	// cases (and helps FF_FB produce k-winner-take-all dynamics), but if average
	// netinputs are lower than typical, you may need to lower it.
	FF0 float32 `default:"0.1"`

	// rate = 1 / tau
	FBDt float32 `edit:"-" display:"-" json:"-" xml:"-"`
}

func (fb *FFFB) Update() {
	fb.FBDt = 1 / fb.FBTau
}

func (fb *FFFB) Defaults() {
	fb.Gi = 1.8
	fb.FF = 1
	fb.FB = 1
	fb.FBTau = 1.4
	fb.MaxVsAvg = 0
	fb.FF0 = 0.1
	fb.Update()
}

func (fb *FFFB) ShouldDisplay(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return fb.On.IsTrue()
	}
}

// FFInhib returns the feedforward inhibition value based on
// average and max excitatory conductance within relevant scope.
func (fb *FFFB) FFInhib(avgGe, maxGe float32) float32 {
	ffNetin := avgGe + fb.MaxVsAvg*(maxGe-avgGe)
	var ffi float32
	if ffNetin > fb.FF0 {
		ffi = fb.FF * (ffNetin - fb.FF0)
	}
	return ffi
}

// FBInhib computes feedback inhibition value as function of average activation
func (fb *FFFB) FBInhib(avgAct float32) float32 {
	return fb.FB * avgAct
}

// FBUpdt updates feedback inhibition using time-integration rate constant
func (fb *FFFB) FBUpdt(fbi float32, newFbi float32) float32 {
	nfb := fbi
	nfb += fb.FBDt * (newFbi - nfb)
	return nfb
}

//gosl:end
