# Sudoku solver

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
