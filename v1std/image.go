// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package v1std

import (
	"image"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/iox/imagex"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorcore"
	"github.com/anthonynsimon/bild/transform"
	"github.com/emer/v1vision/v1vision"
)

//go:generate core generate -add-types

// Image manages conversion of a bitmap image into tensor formats for
// subsequent processing by filters.
type Image struct {

	// File is the name of image file to operate on
	File string

	// Size is the target image size to use. Images will be rescaled to this size.
	Size image.Point

	// Image is the current input image, as Go [image.Image].
	Image image.Image `display:"-"`

	// Tsr is the current input image as an RGB tensor.
	// This points into the V1Vision.Images input image 0.
	Tsr *tensor.Float32 `display:"no-inline"`
}

func (vi *Image) Defaults() {
	vi.Size = image.Point{128, 128}
}

// SetImageResize sets current image for processing, resizing to target size.
func (vi *Image) SetImageResize(img image.Image) {
	vi.Image = img
	isz := vi.Image.Bounds().Size()
	if isz != vi.Size {
		vi.Image = transform.Resize(vi.Image, vi.Size.X, vi.Size.Y, transform.Linear)
	}
}

// OpenImageResize opens image from given filename, and resizes to target size.
func (vi *Image) OpenImageResize(fn string) error {
	img, _, err := imagex.Open(fn)
	if err != nil {
		return errors.Log(err)
	}
	vi.SetImageResize(img)
	return nil
}

// GetTensor gets the Images tensor at given index (typically 0).
func (vi *Image) GetTensor(v1 *v1vision.V1Vision, idx int) {
	vi.Tsr = v1.Images.SubSpace(idx).(*tensor.Float32)
	tensorcore.AddGridStylerTo(vi.Tsr, func(s *tensorcore.GridStyle) {
		s.Image = true
		s.Range.SetMin(0)
	})
}

// SetImageRGB sets current image for processing
// and converts to a float32 tensor with full RGB components.
// border is the border size to add around edges.
func (vi *Image) SetImageRGB(v1 *v1vision.V1Vision, img image.Image, border int) {
	vi.SetImageResize(img)
	vi.GetTensor(v1, 0)
	v1vision.RGBToTensor(vi.Image, vi.Tsr, border, v1vision.BottomZero)
}

// SetImageGrey sets current image for processing
// and converts to a float32 tensor as greyscale image.
// border is the border size to add around edges.
func (vi *Image) SetImageGrey(v1 *v1vision.V1Vision, img image.Image, border int) {
	vi.SetImageResize(img)
	vi.GetTensor(v1, 0)
	v1vision.RGBToGrey(vi.Image, vi.Tsr, border, v1vision.BottomZero)
}
