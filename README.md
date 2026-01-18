# V1 vision

This repository contains visual processing packages in Go (golang), focused on providing efficient V1 (primary visual cortex) level filtering of images, with the output then suitable as input for neural networks.

<p align="center">
    <a href="https://goreportcard.com/report/github.com/emer/v1vision"><img src="https://goreportcard.com/badge/github.com/emer/v1vision" alt="Go Report Card"></a>
    <a href="https://pkg.go.dev/github.com/emer/v1vision"><img src="https://img.shields.io/badge/dev-reference-007d9c?logo=go&logoColor=white&style=flat" alt="pkg.go.dev docs"></a>
    <a href="https://github.com/emer/v1vision/actions"><img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/emer/v1vision/go.yml"></a>
    <a href="https://raw.githack.com/wiki/emer/v1vision/coverage.html"><img alt="Test Coverage" src="https://github.com/emer/v1vision/wiki/coverage.svg"></a>
    <a href="https://github.com/emer/v1vision/tags"><img alt="Version" src="https://img.shields.io/github/v/tag/emer/v1vision?label=version&color=blue"></a>
</p>

As of Dec, 2025, the system supports full GPU-based operations using [GoSL](https://www.cogentcore.org/lab/gosl). See below for design considerations.
 
Two main types of filters are supported:

* **Gabor** filters simulate V1 simple-cell responses in terms of an oriented sine wave times a gaussian envelope that localizes the filter in space. This produces an edge detector that detects oriented contrast transitions between light and dark. In general, the main principle of primary visual filtering is to focus on spatial (and temporal) changes, while filtering out static, uniform areas.

* **DoG** (difference of gaussian) filters simulate retinal On-center vs. Off-center contrast coding cells -- unlike gabor filters, these do not have orientation tuning. Mathematically, they are a difference between a narrow (center) vs wide (surround) gaussian, of opposite signs, balanced so that a uniform input generates offsetting values that sum to zero. In the visual system, orientation tuning is constructed from aligned DoG-like inputs, but it is more efficient to just use the Gabor filters directly. However, DoG filters capture the "blob" cells that encode color contrasts.

The `v1vision` package contains general-purpose filtering code that applies (convolves) any given filter with a visual input. It also supports converting an `image.Image` into a `tensor.Float32` tensor which is the main data type used in this framework. It also supports max-pooling for efficiently reducing the dimensionality of inputs.

The `kwta` package provides an implementation of the feedforward and feedback (FFFB) inhibition dynamics (and noisy X-over-X-plus-1 activation function) from the `Leabra` algorithm to produce a k-Winners-Take-All processing of visual filter outputs -- this increases the contrast and simplifies the representations, and is a good model of the dynamics in primary visual cortex.

To more fully leverage the GPU parallel processing, there is an `NData` data-parallel parameter that runs `n` copies of each operation in parallel. This should correspond to the data-parallel batch size parameter in the simulation, so the entire batch is processed in one step. This parameter must be set at the outset (on `V1Vision` object) to ensure consistent memory allocations for all operations.

## GoSL design

The [GoSL](https://www.cogentcore.org/lab/gosl) (Go as a shader language) system is maximally efficient if everything can be configured statically in memory at the outset, and then each iteration just pushes up the new image and retrieves the final filtered results. This is accomplished by effectively compiling a programmed sequence of operations into the `Ops` list, and configuring everything to hold all the intermediate data results from each Op. At run-time, each Op is uploaded to the GPU in turn, and provides the control params for running that operation.


