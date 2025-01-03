// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kwta

//go:generate core generate -add-types

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
	"github.com/emer/v1vision/fffb"
	"github.com/emer/v1vision/nxx1"
)

// KWTA contains all the parameters needed for computing FFFB
// (feedforward & feedback) inhibition that results in roughly
// k-Winner-Take-All behavior.
type KWTA struct {

	// whether to run kWTA or not
	On bool

	// maximum number of iterations to perform
	Iters int

	// threshold on delta-activation (change in activation) for stopping updating of activations
	DelActThr float32 `default:"0.005"`

	// layer-level feedforward & feedback inhibition -- applied over entire set of values
	LayFFFB fffb.Params `display:"inline"`

	// pool-level (feature groups) feedforward and feedback inhibition -- applied within inner-most dimensions inside outer 2 dimensions (if Pool method is called)
	PoolFFFB fffb.Params `display:"inline"`

	// Noisy X/X+1 rate code activation function parameters
	XX1 nxx1.Params `display:"inline"`

	// time constant for integrating activation
	ActTau float32 `default:"3"`

	// maximal conductances levels for channels
	Gbar Chans `display:"inline"`

	// reversal potentials for each channel
	Erev Chans `display:"inline"`

	// Erev - Act.Thr for each channel -- used in computing GeThrFromG among others
	ErevSubThr Chans `edit:"-" display:"-"`

	// Act.Thr - Erev for each channel -- used in computing GeThrFromG among others
	ThrSubErev Chans `edit:"-" display:"-" json:"-" xml:"-"`

	//
	ActDt float32 `display:"-"; json"-" xml"-" desc:"integration rate = 1/ tau"`
}

func (kwta *KWTA) Defaults() {
	kwta.On = true
	kwta.Iters = 20
	kwta.DelActThr = 0.005
	kwta.LayFFFB.Defaults()
	kwta.PoolFFFB.Defaults()
	kwta.LayFFFB.On = true
	kwta.PoolFFFB.On = true
	kwta.PoolFFFB.Gi = 2.0
	kwta.XX1.Defaults()
	kwta.ActTau = 3
	kwta.Gbar.SetAll(0.5, 0.1, 1.0, 1.0) // 0.5 is key for 1.0 inputs
	kwta.Erev.SetAll(1.0, 0.3, 0.3, 0.1)
	kwta.Update()
}

// Update must be called after any changes to parameters
func (kwta *KWTA) Update() {
	kwta.LayFFFB.Update()
	kwta.PoolFFFB.Update()
	kwta.XX1.Update()
	kwta.ErevSubThr.SetFromOtherMinus(kwta.Erev, kwta.XX1.Thr)
	kwta.ThrSubErev.SetFromMinusOther(kwta.XX1.Thr, kwta.Erev)
	kwta.ActDt = 1 / kwta.ActTau
}

// GeThrFromG computes the threshold for Ge based on other conductances
func (kwta *KWTA) GeThrFromG(gi float32) float32 {
	ge := ((kwta.Gbar.I*gi*kwta.ErevSubThr.I + kwta.Gbar.L*kwta.ErevSubThr.L) / kwta.ThrSubErev.E)
	return ge
}

// ActFromG computes rate-coded activation Act from conductances Ge and Gi
func (kwta *KWTA) ActFromG(geThr, ge, act float32) (nwAct, delAct float32) {
	nwAct = kwta.XX1.NoisyXX1(ge*kwta.Gbar.E - geThr)
	delAct = kwta.ActDt * (nwAct - act)
	nwAct = act + delAct
	return nwAct, delAct
}

// KWTALayer computes k-Winner-Take-All activation values from raw inputs.
// act output tensor is set to same shape as raw inputs if not already.
// This version just computes a "layer" level of inhibition across the
// entire set of tensor values.
// extGi is extra / external Gi inhibition per unit
// -- e.g. from neighbor inhib -- must be size of raw, act.
func (kwta *KWTA) KWTALayer(raw, act, extGi *tensor.Float32) {
	inhib := fffb.Inhib{}
	raws := raw.Values // these are ge

	act.SetShapeSizes(raw.Shape().Sizes...)
	if extGi != nil {
		extGi.SetShapeSizes(raw.Shape().Sizes...)
	}

	acts := act.Values

	inhib.Ge.Init()
	for i, ge := range raws {
		inhib.Ge.UpdateValue(ge, int32(i))
	}
	inhib.Ge.CalcAvg()

	for cy := 0; cy < kwta.Iters; cy++ {
		kwta.LayFFFB.Inhib(&inhib)
		inhib.Act.Init()
		maxDelAct := float32(0)
		for i := range acts {
			gi := inhib.Gi
			if extGi != nil {
				gi += extGi.Values[i]
			}
			geThr := kwta.GeThrFromG(gi)
			ge := raws[i]
			nwAct, delAct := kwta.ActFromG(geThr, ge, acts[i])
			maxDelAct = math32.Max(maxDelAct, math32.Abs(delAct))
			inhib.Act.UpdateValue(nwAct, int32(i))
			acts[i] = nwAct
		}
		inhib.Act.CalcAvg()
		if cy > 2 && maxDelAct < kwta.DelActThr {
			break
		}
	}
}

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
func (kwta *KWTA) KWTAPool(raw, act *tensor.Float32, inhib *fffb.Inhibs, extGi *tensor.Float32) {
	layInhib := fffb.Inhib{}

	raws := raw.Values // these are ge

	act.SetShapeSizes(raw.Shape().Sizes...)
	if extGi != nil {
		extGi.SetShapeSizes(raw.Shape().Sizes...)
	}

	acts := act.Values

	layY := raw.DimSize(0)
	layX := raw.DimSize(1)
	layN := layY * layX

	plY := raw.DimSize(2)
	plX := raw.DimSize(3)
	plN := plY * plX

	if len(*inhib) < layN {
		if cap(*inhib) < layN {
			*inhib = make([]fffb.Inhib, layN)
		} else {
			*inhib = (*inhib)[0:layN]
		}
	}

	layInhib.Ge.Init()
	pi := 0
	for ly := 0; ly < layY; ly++ {
		for lx := 0; lx < layX; lx++ {
			plInhib := &((*inhib)[pi])
			plInhib.Ge.Init()
			pui := pi * plN
			ui := 0
			for py := 0; py < plY; py++ {
				for px := 0; px < plX; px++ {
					idx := pui + ui
					ge := raws[idx]
					layInhib.Ge.UpdateValue(ge, int32(idx))
					plInhib.Ge.UpdateValue(ge, int32(ui))
					ui++
				}
			}
			plInhib.Ge.CalcAvg()
			pi++
		}
	}
	layInhib.Ge.CalcAvg()

	for cy := 0; cy < kwta.Iters; cy++ {
		kwta.LayFFFB.Inhib(&layInhib)

		layInhib.Act.Init()
		maxDelAct := float32(0)
		pi := 0
		for ly := 0; ly < layY; ly++ {
			for lx := 0; lx < layX; lx++ {
				plInhib := &((*inhib)[pi])

				kwta.PoolFFFB.Inhib(plInhib)

				giPool := math32.Max(layInhib.Gi, plInhib.Gi)

				plInhib.Act.Init()
				pui := pi * plN
				ui := 0
				for py := 0; py < plY; py++ {
					for px := 0; px < plX; px++ {
						idx := pui + ui
						gi := giPool
						if extGi != nil {
							eIn := extGi.Values[idx]
							eGi := kwta.PoolFFFB.Gi * kwta.PoolFFFB.FFInhib(eIn, eIn)
							gi = math32.Max(gi, eGi)
						}
						geThr := kwta.GeThrFromG(gi)
						ge := raws[idx]
						act := acts[idx]
						nwAct, delAct := kwta.ActFromG(geThr, ge, act)
						maxDelAct = math32.Max(maxDelAct, math32.Abs(delAct))
						layInhib.Act.UpdateValue(nwAct, int32(idx))
						plInhib.Act.UpdateValue(nwAct, int32(ui))
						acts[idx] = nwAct

						ui++
					}
				}
				plInhib.Act.CalcAvg()
				pi++
			}
		}
		layInhib.Act.CalcAvg()
		if cy > 2 && maxDelAct < kwta.DelActThr {
			// fmt.Printf("under thr at cycle: %v\n", cy)
			break
		}
	}
}
