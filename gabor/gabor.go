// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
package gabor provides a gabor filter for visual and other
forms of signal processing
*/
package gabor

//go:generate core generate -add-types -gosl

import (
	"math"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
)

// gabor.Filter specifies a gabor filter function,
// i.e., a 2d Gaussian envelope times a sinusoidal plane wave.
// By default it produces 2 phase asymmetric edge detector filters.
type Filter struct {

	// On is whether this filter active.
	On bool

	// Gain is the overall gain multiplier applied after filtering.
	// Only relevant if not using renormalization
	// (otherwize it just gets renormed away).
	Gain float32 `default:"2"`

	// Size of the overall filter, which is the  number of pixels
	// wide and tall for a square matrix used to encode the filter.
	// Filter is centered within this square, typically an even number,
	//  min effective size ~6.
	Size int

	// Wavelength is the wavelength of the sine waves: number of pixels
	// over which a full period of the wave takes place.
	// Typically same as Size (computation adds a 2 PI factor to
	// translate into pixels instead of radians).
	Wavelength float32

	// Spacing is how far apart to space the centers of the
	// gabor filters. 1 = every pixel, 2 = every other pixel,
	// etc. High-res should be 1 or 2; lower res can be increments
	// therefrom.
	Spacing int

	// SigmaLen is the Gaussian sigma for the length dimension
	// (elongated axis perpendicular to the sine waves).
	// as a normalized proportion of filter Size.
	SigmaLen float32 `default:"0.3"`

	// SigmaWd is the gaussian sigma for the width dimension
	// (in the direction of the sine waves), as a normalized proportion
	// of filter size.
	SigmaWd float32 `default:"0.15,0.2"`

	// Phase offset for the sine wave, in degrees.
	// 0 = asymmetric sine wave, 90 = symmetric cosine wave.
	Phase float32 `default:"0,90"`

	// CircleEdge cuts off the filter (to zero) outside a circle
	// of diameter = Size. Makes the filter more radially symmetric.
	CircleEdge bool `default:"true"`

	// NAngles is the number of different angles of overall gabor
	// filter orientation to use. First angle is always horizontal.
	NAngles int `default:"4"`
}

func (gf *Filter) Defaults() {
	gf.On = true
	gf.Gain = 2
	gf.Size = 6
	gf.Spacing = 2
	gf.Wavelength = 6
	gf.SigmaLen = 0.3
	gf.SigmaWd = 0.2
	gf.Phase = 0
	gf.CircleEdge = true
	gf.NAngles = 4
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

// SetSize sets the size and Wavelength to same value, and also sets spacing
// these are the main params that need to be varied for standard V1 gabors
func (gf *Filter) SetSize(sz, spc int) {
	gf.Size = sz
	gf.Wavelength = float32(sz)
	gf.Spacing = spc
}

// ToTensor renders filters into the given tensor.Tensor.
// must have dimensions already set to [angle][Y][X] where Y = X = Size
func (gf *Filter) ToTensor(tsr *tensor.Float32) {
	ctr := 0.5 * float32(gf.Size-1)
	angInc := math.Pi / float32(gf.NAngles)

	radius := float32(gf.Size) * 0.5

	gsLen := gf.SigmaLen * float32(gf.Size)
	gsWd := gf.SigmaWd * float32(gf.Size)

	lenNorm := 1.0 / (2.0 * gsLen * gsLen)
	wdNorm := 1.0 / (2.0 * gsWd * gsWd)

	twoPiNorm := (2.0 * math.Pi) / gf.Wavelength
	phsRad := math32.DegToRad(gf.Phase)

	for ang := 0; ang < gf.NAngles; ang++ {
		angf := -float32(ang) * angInc

		posSum := float32(0)
		negSum := float32(0)
		for x := 0; x < gf.Size; x++ {
			for y := 0; y < gf.Size; y++ {
				xf := float32(x) - ctr
				yf := float32(y) - ctr

				dist := math32.Hypot(xf, yf)
				val := float32(0)
				if !(gf.CircleEdge && (dist > radius)) {
					nx := xf*math32.Cos(angf) - yf*math32.Sin(angf)
					ny := yf*math32.Cos(angf) + xf*math32.Sin(angf)
					gauss := math32.Exp(-(lenNorm*(nx*nx) + wdNorm*(ny*ny)))
					sin := math32.Sin(twoPiNorm*ny + phsRad)
					val = gauss * sin
					if val > 0 {
						posSum += val
					} else if val < 0 {
						negSum += -val
					}
				}
				tsr.Set(val, ang, y, x)
			}
		}
		// renorm each half
		posNorm := float32(1) / posSum
		negNorm := float32(1) / negSum
		for x := 0; x < gf.Size; x++ {
			for y := 0; y < gf.Size; y++ {
				val := tsr.Value(ang, y, x)
				if val > 0 {
					val *= posNorm
				} else if val < 0 {
					val *= negNorm
				}
				tsr.Set(val, ang, y, x)
			}
		}
	}
}

// ToTable renders filters into the given table.Table
// setting a column named Angle to the angle and
// a column named Gabor to the filter for that angle.
// This is useful for display and validation purposes.
func (gf *Filter) ToTable(tab *table.Table) {
	tab.AddFloat32Column("Angle")
	tab.AddFloat32Column("Filter", gf.Size, gf.Size)
	tab.SetNumRows(gf.NAngles)
	cl := tab.Columns.Values[1].(*tensor.Float32)
	gf.ToTensor(cl)
	angInc := math.Pi / float32(gf.NAngles)
	for ang := 0; ang < gf.NAngles; ang++ {
		angf := math32.RadToDeg(-float32(ang) * angInc)
		tab.ColumnByIndex(0).SetFloat1D(float64(-angf), ang)
	}
}
