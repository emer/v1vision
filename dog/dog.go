// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
package dog provides the Difference-of-Gaussians (DoG) filter for visual and other
forms of signal processing
*/
package dog

//go:generate core generate -add-types -gosl

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
)

// dog.Filter specifies a DoG Difference of Gaussians filter function.
// On = narrower on-center peak; Off = broader surround;
// Net = On - Off difference that is selectively activated by differential
// On vs. Off activity, and is silent for any consistent uniform field.
// Can also be used for color contrast filtering with same-sized On and Off
// Gaussians, applied to different color (RGB / LMS) channels.
type Filter struct {

	// On is whether this filter is active.
	On bool

	// Gain is the overall gain multiplier applied after dog filtering.
	// Only relevant if not using renormalization on the output
	// (otherwize it just gets renormed away).
	Gain float32 `default:"8"`

	// OnGain applies only to the on component of filter,
	// which is only relevant for color contrast DoG's.
	OnGain float32 `default:"1"`

	// Size of the overall filter, which is the number of pixels wide
	// and tall for a square matrix used to encode the filter.
	// The filter is centered within this square.
	// Typically an even number, min effective size ~6.
	Size int

	// Spacing is how far apart to space the centers of the dog filters.
	// 1 = every pixel, 2 = every other pixel, etc.
	// high-res should be 1 or 2, lower res can be increments therefrom.
	Spacing int

	// OnSigma is the Gaussian sigma for the narrower On gaussian,
	// in normalized units relative to Size.
	OnSigma float32 `default:"0.125"`

	// OffSigma is the Gaussian sigma for the wider Off gaussian,
	// in normalized units relative to Size.
	OffSigma float32 `default:"0.25"`

	// CircleEdge cuts off the filter (to zero) outside a circle of diameter
	//  = Size. Makes the filter more radially symmetric.
	CircleEdge bool `default:"true"`
}

func (gf *Filter) Defaults() {
	gf.On = true
	gf.Gain = 8
	gf.OnGain = 1
	gf.Size = 12
	gf.Spacing = 4
	gf.OnSigma = 0.125
	gf.OffSigma = 0.25
	gf.CircleEdge = true
}

func (gf *Filter) Update() {
}

func (gf *Filter) ShouldDisplay(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return gf.On
	}
}

// SetSize sets the size and spacing -- these are the main params
// that need to be varied for standard V1 dogs.
func (gf *Filter) SetSize(sz, spc int) {
	gf.Size = sz
	gf.Spacing = spc
}

// GaussDenSigma returns gaussian density for given value and sigma
func GaussDenSigma(x, sig float32) float32 {
	x /= sig
	return 0.398942280 * math32.Exp(-0.5*x*x) / sig
}

// SetSameSigma sets the On and Off sigma to the same value,
// for e.g., color contrast filtering instead of spatial on/off filtering.
// A value of 0.5 is typically used, to obtain more spatial coverage for
// broader "blob" tuning instead of spatial filtering.
func (gf *Filter) SetSameSigma(sigma float32) {
	gf.OnSigma = sigma
	gf.OffSigma = sigma
}

//	tsr.SetShapeSizes(int(FiltersN), gf.Size, gf.Size)

// ToTensor renders dog filter into the given tensor.Tensor, which has
// 3 dimensions: FilterNo, Y, X, where Y = X = Size.
// The specified list of filters is written in given order.
func (gf *Filter) ToTensor(tsr *tensor.Float32, filters ...Filters) {
	ctr := 0.5 * float32(gf.Size-1)
	radius := float32(gf.Size) * 0.5

	gsOn := gf.OnSigma * float32(gf.Size)
	gsOff := gf.OffSigma * float32(gf.Size)

	var idxs [FiltersN]int
	for i := range FiltersN {
		idxs[i] = -1
	}
	for i, fl := range filters {
		idxs[fl] = i
	}

	var posSum, negSum, onSum, offSum float32
	for y := 0; y < gf.Size; y++ {
		for x := 0; x < gf.Size; x++ {
			xf := float32(x) - ctr
			yf := float32(y) - ctr

			dist := math32.Hypot(xf, yf)
			var ong, offg float32
			if !(gf.CircleEdge && (dist > radius)) {
				ong = GaussDenSigma(dist, gsOn)
				offg = GaussDenSigma(dist, gsOff)
			}
			net := ong - offg
			if net > 0 {
				posSum += net
			} else if net < 0 {
				negSum += -net
			}
			onSum += ong
			offSum += offg
			if fi := idxs[Net]; fi >= 0 {
				tsr.Set(net, fi, y, x)
			}
			if fi := idxs[On]; fi >= 0 {
				tsr.Set(ong, fi, y, x)
			}
			if fi := idxs[Off]; fi >= 0 {
				tsr.Set(offg, fi, y, x)
			}
		}
	}
	// renorm each half, separate components
	for y := 0; y < gf.Size; y++ {
		for x := 0; x < gf.Size; x++ {
			if fi := idxs[Net]; fi >= 0 {
				val := tsr.Value(fi, y, x)
				if val > 0 {
					val /= posSum
				} else if val < 0 {
					val /= negSum
				}
				tsr.Set(val, fi, y, x)
			}
			if fi := idxs[On]; fi >= 0 {
				on := tsr.Value(fi, y, x)
				tsr.Set(on/onSum, fi, y, x)
			}
			if fi := idxs[Off]; fi >= 0 {
				off := tsr.Value(fi, y, x)
				tsr.Set(off/offSum, fi, y, x)
			}
		}
	}
}

// ToTable renders filters into the given table.Table
// setting a column named Version and  a column named Filter
// to the filter for that version (on, off, net)
// This is useful for display and validation purposes.
func (gf *Filter) ToTable(tab *table.Table) {
	tab.AddStringColumn("Version")
	tab.AddFloat32Column("Filter", gf.Size, gf.Size)
	tab.SetNumRows(3)
	cl := tab.Columns.Values[1].(*tensor.Float32)
	gf.ToTensor(cl, On, Off, Net)
	nm := tab.ColumnByIndex(0)
	for fn := range FiltersN {
		nm.SetString(fn.String(), int(fn))
	}
}

// Filters is the type of filter.
type Filters int32 //enums:enum

const (
	// On is the (by convention) smaller on-center peak filter.
	On Filters = iota

	// Off is the larger off-center surround filter.
	Off

	// Net is On - Off, separately normalized so that application to
	// a uniform field results in a zero. This is for spatial contrast
	// filtering.
	Net
)
