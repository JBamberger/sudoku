//
// Created by jannik on 17.04.2021.
//

#ifndef CONFIG_H
#define CONFIG_H

#include <filesystem>
namespace fs = std::filesystem;

const fs::path rootPath = fs::current_path().parent_path().parent_path();

const fs::path sharePath = rootPath / "share";
const fs::path dataPath = rootPath / "data";

const fs::path digits5kDatasetPath = dataPath / "digits.png";

const fs::path digitDatasetPath = dataPath / "digit_dataset";
const fs::path extractedDigitsPath = dataPath / "extractedDigits";
const fs::path digitSamplePath = dataPath / "digit_smaples";

const fs::path sudokusPath = dataPath / "sudokus";
const fs::path sudokusGtPath = sudokusPath / "ground_truth_new.csv";

const fs::path classifierCheckpointPath = sharePath / "digit_classifier.pth";


#endif // CONFIG_H
