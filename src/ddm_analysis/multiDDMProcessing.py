import cupy

# Some custom Cupy functions that go faster than just writing numpy ------------------

# Calculates the Fourier transform at (sampleX, sampleY) of a
# delta function at (positionX, positionY).
# remember: Fourier(delta_(x,y))(q_x,q_y) = exp(i(x*q_x+y*q_y))


@cupy.fuse()
def deltaTransform(positionX, positionY, sampleX, sampleY):
    return cupy.exp(1j * (positionX * sampleX + positionY * sampleY), dtype=cupy.complex64)

# Faster modulus squared


@cupy.fuse()
def modSquare(differences):
    return cupy.square(cupy.abs(differences))


# Other helper functions -----------------------------------------------------------------

# Works out what box every particle is in for every frame.


def assignBoxNumbers(positions, frameSize, gridSize):
    boxSize = frameSize / gridSize
    boxCoordinates = positions / boxSize
    boxCoordinates = cupy.where(boxCoordinates == 0, 0.000000001, boxCoordinates)
    boxCoordinates = cupy.where(
        boxCoordinates == cupy.round(-cupy.abs(boxCoordinates)),
        boxCoordinates + 1,
        boxCoordinates,
    )
    boxCoordinates = cupy.where(
        boxCoordinates > 0,
        boxCoordinates.astype(cupy.int32),
        boxCoordinates.astype(cupy.int32) - 1,
    )
    return boxCoordinates


# multi-DDM processing routines -------------------------------------------------------------


def doBaseTransform(particlePositions, outputPositions, gridSize, frameSize):
    """
    Does the Fourier transform step using cupy.add.at on separate real and
    imaginary components to support complex atomic additions.
    """
    # Find what box every particle is in on each frame.
    boxNumbers = assignBoxNumbers(particlePositions, frameSize, gridSize)

    # Get some parameters from the array sizes.
    numberOfFrames = particlePositions.shape[0]
    numberOfOutputGroups = outputPositions.shape[0]
    numberOfOutputsPerGroup = outputPositions.shape[1]

    # Extract the x and y coordinates.
    particlesX = particlePositions[:, :, 0]
    particlesY = particlePositions[:, :, 1]
    outputsX = outputPositions[:, :, 0]
    outputsY = outputPositions[:, :, 1]

    # Create separate real and imaginary arrays for the output,
    # as cupy.add.at does not support complex types directly.
    boxTransforms_real = cupy.zeros(
        (
            numberOfFrames,
            gridSize,
            gridSize,
            numberOfOutputGroups,
            numberOfOutputsPerGroup,
        ),
        dtype=cupy.float32,
    )
    boxTransforms_imag = cupy.zeros(
        (
            numberOfFrames,
            gridSize,
            gridSize,
            numberOfOutputGroups,
            numberOfOutputsPerGroup,
        ),
        dtype=cupy.float32,
    )

    # Do transform for each frame one at a time.
    for frameNumber in range(numberOfFrames):
        # Calculate Fourier transform of each particle's delta function at all output positions.
        individualParticleTransforms = deltaTransform(
            particlesX[frameNumber, :, None, None],
            particlesY[frameNumber, :, None, None],
            outputsX[None, :, :],
            outputsY[None, :, :],
        )

        # Get the box coordinates for each particle in the current frame
        box_x = boxNumbers[frameNumber, :, 0]
        box_y = boxNumbers[frameNumber, :, 1]

        # Perform atomic add on real and imaginary parts separately.
        cupy.add.at(
            boxTransforms_real[frameNumber],
            (box_x, box_y),
            individualParticleTransforms.real,
        )
        cupy.add.at(
            boxTransforms_imag[frameNumber],
            (box_x, box_y),
            individualParticleTransforms.imag,
        )

    # Wait for GPU to finish
    cupy.cuda.stream.get_current_stream().synchronize()

    # Combine the real and imaginary parts back into a complex array.
    boxTransforms = boxTransforms_real + 1j * boxTransforms_imag

    del boxTransforms_real
    del boxTransforms_imag

    return boxTransforms


# Takes the transform for the smallest boxes and uses it to generate multi-DDM
# output for them or a larger box size.


def getDDMOutput(baseTransform, blockSize, taus):

    # If block size is 1 then we are working with the correct box sizes,
    # if not then need to generate date for the correct box size from what we have.
    if blockSize != 1:
        transform = collapseBaseTransform(baseTransform, blockSize)
    else:
        transform = baseTransform

    # Get parameters from array shapes.
    numberOfTaus = taus.shape[0]
    gridSize = transform.shape[1]
    numberOfOutputGroups = transform.shape[3]

    # Set up output array to write data into.
    transformData = cupy.zeros(
        (numberOfTaus, gridSize, gridSize, numberOfOutputGroups), cupy.float32
    )

    # For each delay subtract all pairs of frames then mod square result and average
    # across all pairs and grouped output positions.
    for tauIndex in range(len(taus)):
        tau = int(taus[tauIndex])
        differences = transform[:-tau] - transform[tau:]
        transformData[tauIndex] = cupy.mean(modSquare(differences), axis=(0, -1))

    # Wait for GPU to finish
    cupy.cuda.stream.get_current_stream().synchronize()

    # Swap axes around because it's more intuitive this way: (gridX, gridY, Fourier direction, delay)
    transformData = cupy.moveaxis(transformData, 0, 3)

    return transformData


# Adds the Fourier transforms for the smallest boxes together to get the result for
# larger box sizes.


def collapseBaseTransform(baseTransform, blockSize):

    # Get parameters from array shapes.
    numberOfFrames = baseTransform.shape[0]
    newGridSize = int(baseTransform.shape[1] / blockSize)
    numberOfOutputGroups = baseTransform.shape[3]
    numberOfOutputsPerGroup = baseTransform.shape[4]

    # Crop boxes that wont fit into new grid.
    transforms = baseTransform[
        :, : newGridSize * blockSize, : newGridSize * blockSize, :
    ]

    # Sum the correct small boxes together to get the appropriate larger boxes.
    transforms = transforms.reshape(
        (
            numberOfFrames,
            newGridSize,
            blockSize,
            newGridSize,
            blockSize,
            numberOfOutputGroups,
            numberOfOutputsPerGroup,
        )
    )
    transforms = cupy.sum(transforms, axis=(2, 4))

    return transforms
