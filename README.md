# Sudoku solver

The repository contains code for a sudoku solving application for Android. The sudoku is detected in
the live camera feed and solved. The solution is then superimposed over the image, giving an AR-like
look.

Further, the repository contains the prototypical implementation of the algorithms in python.

The sudoku detection uses standard OpenCV functions to find the sudoku outline and cells. Each cell
is classified with a small CNN. The sudoku solver uses a _Dancing Links_ implementation.

The Android application is very un-optimized and performs the processing on the main thread. This
leads to lagging UI updated etc. Further, it does not correctly query the sensor orientation, thus
the image preview has the wrong orientation on some phones.

## Outdated information

Idea:

1. Read an image file / take an image.
2. Detect the sudoku within.
3. Solve the sudoku.
4. Render the solution into the image.

Current problems:

- [x] Sudoku detection is not complete
- [x] Digit classifier can be trained on MNIST but there is no input normalization therefore the classifier does not work.
- [ ] The classifier has a very low accuracy.
- [ ] Hardcoded paths
- [x] No sudoku solver
- [ ] The entire pipeline is probably not very robust and there are no tests.
- [ ] Probably a lot more
