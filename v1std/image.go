// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1std

import (
	"image"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/lab/tensor"
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/v1vision/v1vision"
)

//go:generate core generate -add-types

// Image manages conversion of bitmap images into tensor formats for
// subsequent processing by filters.
type Image struct {

	// File is the name of image file to operate on
	File string

	// Size is the target image size to use. Images will be rescaled to this size.
	Size image.Point

	// Images are the current input image(s), as Go [image.Image].
	Images []image.Image `display:"-"`

	// Tsr are the current input image(s) as an RGB tensor.
	// This points into the V1Vision.Images input image.
	Tsr *tensor.Float32 `display:"no-inline"`
}

func (vi *Image) Defaults() {
	vi.Size = image.Point{128, 128}
}

// SetImagesResize sets current image(s) for processing, resizing to target size.
func (vi *Image) SetImagesResize(imgs ...image.Image) {
	// todo: do this all on GPU at some point!
	vi.Images = imgs
	for i, im := range vi.Images {
		isz := im.Bounds().Size()
		if isz != vi.Size {
			vi.Images[i] = transform.Resize(im, vi.Size.X, vi.Size.Y, transform.Linear)
		}
	}
}

// OpenImagesResize opens image(s) from given filename(s), and resizes to target size.
func (vi *Image) OpenImagesResize(fns ...string) error {
	var errs []error
	imgs := make([]image.Image, len(fns))
	for i, fn := range fns {
		img, _, err := imagex.Open(fn)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		imgs[i] = img
	}
	vi.SetImagesResize(imgs...)
	return errors.Join(errs...)
}

// GetTensors gets the Images tensor at given index (typically 0).
func (vi *Image) GetTensors(v1 *v1vision.V1Vision, idx int) {
	vi.Tsr = v1.Images.SubSpace(idx).(*tensor.Float32)
	// todo:
	// tensorcore.AddGridStylerTo(vi.Tsr, func(s *tensorcore.GridStyle) {
	// 	s.Image = true
	// 	s.Range.SetMin(0)
	// })
}

// SetImagesRGB sets current image(s) for processing
// and converts to a float32 tensor with full RGB components.
// border is the border size to add around edges.
func (vi *Image) SetImagesRGB(v1 *v1vision.V1Vision, border int, imgs ...image.Image) {
	vi.SetImagesResize(imgs...)
	vi.GetTensors(v1, 0)
	v1vision.RGBToTensor(vi.Tsr, border, v1vision.BottomZero, vi.Images...)
}

// SetImagesGrey sets current image(s) for processing
// and converts to a float32 tensor as greyscale image.
// border is the border size to add around edges.
func (vi *Image) SetImagesGrey(v1 *v1vision.V1Vision, border int, imgs ...image.Image) {
	vi.SetImagesResize(imgs...)
	vi.GetTensors(v1, 0)
	v1vision.RGBToGrey(vi.Tsr, border, v1vision.BottomZero, vi.Images...)
}
