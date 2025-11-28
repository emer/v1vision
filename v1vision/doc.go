// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package v1vision provides filtering methods for the v1vision package.
These apply tensor.Tensor filters to a 2D visual input, via Convolve
(convolution) function. Other more advanced filters are supported too.

Full GPU-based computation is supported via the https://cogentcore.org/lab/GoSL
Go-as-a-shading-language system. This is also very efficient on CPU because
everything has been organized in a maximally efficient parallel manner.

image.go contains routines for converting an image into the float32
tensor.Float32 that is required for doing the convolution.
* RGBToGrey converts an RGB image to a greyscale float32.

MaxPool function does Max-pooling over filtered results to reduce
dimensionality, consistent with standard DCNN approaches.

Geom manages the geometry for going from an input image to the
filtered output of that image.

For maximum efficiency, all input images must be padded so that the filters
can be applied directly without any range checking. There are support functions
to add appropriate padding borders (e.g., WrapPad).
*/
package v1vision
