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

	// whether to run kWTA or not
	On slbool.Bool

	// maximum number of iterations to perform
	Iters int32

	// threshold on delta-activation (change in activation) for stopping updating of activations
	DelActThr float32 `default:"0.005"`

	// time constant for integrating activation
	ActTau float32 `default:"3"`

	// layer-level feedforward & feedback inhibition -- applied over entire set of values
	LayFFFB fffb.FFFB `display:"inline"`

	// pool-level (feature groups) feedforward and feedback inhibition -- applied within inner-most dimensions inside outer 2 dimensions (if Pool method is called)
	PoolFFFB fffb.FFFB `display:"inline"`

	// Noisy X/X+1 rate code activation function parameters
	XX1 nxx1.Params `display:"inline"`

	// maximal conductances levels for channels
	Gbar Chans `display:"inline"`

	// reversal potentials for each channel
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
	kp.Iters = 20
	kp.DelActThr = 0.005
	kp.LayFFFB.Defaults()
	kp.PoolFFFB.Defaults()
	kp.LayFFFB.On.SetBool(true)
	kp.PoolFFFB.On.SetBool(true)
	kp.PoolFFFB.Gi = 2.0
	kp.XX1.Defaults()
	kp.ActTau = 3
	kp.Gbar.SetAll(0.5, 0.1, 1.0, 1.0) // 0.5 is key for 1.0 inputs
	kp.Erev.SetAll(1.0, 0.3, 0.3, 0.1)
	kp.Update()
}

// Update must be called after any changes to parameters
func (kp *KWTA) Update() {
	kp.LayFFFB.Update()
	kp.PoolFFFB.Update()
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

// KWTAPool computes k-Winner-Take-All activation values from raw inputs
// act output tensor is set to same shape as raw inputs if not already.
// This version computes both Layer and Pool (feature-group) level
// inibition -- tensors must be 4 dimensional -- outer 2D is Y, X Layer
// and inner 2D are features (pools) per location.
// The inhib slice is required for pool-level inhibition and will
// be automatically sized to outer X,Y dims if not big enough.
// For best performance store this and reuse to avoid memory allocations.
// extGi is extra / external Gi inhibition per unit
// -- e.g. from neighbor inhib -- must be size of raw, act.
// func (kwta *KWTA) KWTAPool(raw, act *tensor.Float32, inhib *fffb.Inhibs, extGi *tensor.Float32) {
// 	layInhib := fffb.Inhib{}
//
// 	raws := raw.Values // these are ge
//
// 	act.SetShapeSizes(raw.Shape().Sizes...)
// 	if extGi != nil {
// 		extGi.SetShapeSizes(raw.Shape().Sizes...)
// 	}
//
// 	acts := act.Values
//
// 	layY := raw.DimSize(0)
// 	layX := raw.DimSize(1)
// 	layN := layY * layX
//
// 	plY := raw.DimSize(2)
// 	plX := raw.DimSize(3)
// 	plN := plY * plX
//
// 	if len(*inhib) < layN {
// 		if cap(*inhib) < layN {
// 			*inhib = make([]fffb.Inhib, layN)
// 		} else {
// 			*inhib = (*inhib)[0:layN]
// 		}
// 	}
//
// 	layInhib.Ge.Init()
// 	pi := 0
// 	for ly := 0; ly < layY; ly++ {
// 		for lx := 0; lx < layX; lx++ {
// 			plInhib := &((*inhib)[pi])
// 			plInhib.Ge.Init()
// 			pui := pi * plN
// 			ui := 0
// 			for py := 0; py < plY; py++ {
// 				for px := 0; px < plX; px++ {
// 					idx := pui + ui
// 					ge := raws[idx]
// 					layInhib.Ge.UpdateValue(ge, int32(idx))
// 					plInhib.Ge.UpdateValue(ge, int32(ui))
// 					ui++
// 				}
// 			}
// 			plInhib.Ge.CalcAvg()
// 			pi++
// 		}
// 	}
// 	layInhib.Ge.CalcAvg()
//
// 	for cy := 0; cy < kwta.Iters; cy++ {
// 		kwta.LayFFFB.Inhib(&layInhib)
//
// 		layInhib.Act.Init()
// 		maxDelAct := float32(0)
// 		pi := 0
// 		for ly := 0; ly < layY; ly++ {
// 			for lx := 0; lx < layX; lx++ {
// 				plInhib := &((*inhib)[pi])
//
// 				kwta.PoolFFFB.Inhib(plInhib)
//
// 				giPool := math32.Max(layInhib.Gi, plInhib.Gi)
//
// 				plInhib.Act.Init()
// 				pui := pi * plN
// 				ui := 0
// 				for py := 0; py < plY; py++ {
// 					for px := 0; px < plX; px++ {
// 						idx := pui + ui
// 						gi := giPool
// 						if extGi != nil {
// 							eIn := extGi.Values[idx]
// 							eGi := kwta.PoolFFFB.Gi * kwta.PoolFFFB.FFInhib(eIn, eIn)
// 							gi = math32.Max(gi, eGi)
// 						}
// 						geThr := kwta.GeThrFromG(gi)
// 						ge := raws[idx]
// 						act := acts[idx]
// 						nwAct, delAct := kwta.ActFromG(geThr, ge, act)
// 						maxDelAct = math32.Max(maxDelAct, math32.Abs(delAct))
// 						layInhib.Act.UpdateValue(nwAct, int32(idx))
// 						plInhib.Act.UpdateValue(nwAct, int32(ui))
// 						acts[idx] = nwAct
//
// 						ui++
// 					}
// 				}
// 				plInhib.Act.CalcAvg()
// 				pi++
// 			}
// 		}
// 		layInhib.Act.CalcAvg()
// 		if cy > 2 && maxDelAct < kwta.DelActThr {
// 			// fmt.Printf("under thr at cycle: %v\n", cy)
// 			break
// 		}
// 	}
// }
