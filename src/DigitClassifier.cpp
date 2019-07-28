#include "DigitClassifier.h"

void DigitClassifier::train() {
    std::cout << "begin training" << std::endl;
    Mat train_data_mat, train_label_mat;
    Mat test_data_mat, test_label_mat;

    loadMNIST(R"(D:\datasets\mnist\train-images.idx3-ubyte)", R"(D:\datasets\mnist\train-labels.idx1-ubyte)",
              train_data_mat, train_label_mat);

    train_data_mat.convertTo(train_data_mat, CV_32FC1);
    train_label_mat.convertTo(train_label_mat, CV_32SC1);

    svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 500, 1e-6));
    svm->train(train_data_mat, ml::SampleTypes::ROW_SAMPLE, train_label_mat);
    svm->save("SVM_MNIST.xml");
}

DigitClassifier::DigitClassifier() {
    try {
        svm = Algorithm::load<ml::SVM>("SVM_MNIST.xml");
    } catch (const Exception &ex) {
        std::cout << "Error: " << ex.what() << std::endl;
        train();
    }
}

DigitClassifier::~DigitClassifier() {
    delete svm;
}

int reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255u;
    ch2 = (i >> 8u) & 255u;
    ch3 = (i >> 16u) & 255u;
    ch4 = (i >> 24u) & 255u;

    return ((int) ch1 << 24u) + ((int) ch2 << 16u) + ((int) ch3 << 8u) + ch4;
}

bool DigitClassifier::loadMNIST(const std::string &pic_filename, const std::string &label_filename,
                                Mat &training_data, Mat &label_data) {
    std::ifstream pic_file(pic_filename, std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::binary);

    if (!pic_file.is_open() || !label_file.is_open()) {
        return false;
    }

    int magic_number = 0;
    int N = 0;
    int n_rows = 0;
    int n_cols = 0;

    label_file.read((char *) &magic_number, sizeof(magic_number));
    pic_file.read((char *) &magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    label_file.read((char *) &N, sizeof(N));
    pic_file.read((char *) &N, sizeof(N));
    N = reverseInt(N);

    pic_file.read((char *) &n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    pic_file.read((char *) &n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    int imgSize = n_cols * n_rows;
    training_data = Mat(N, imgSize, CV_8U);
    label_data = Mat(N, 1, CV_8U);

    for (int i = 0; i < N; ++i) {
        unsigned char buffer[28*28];
        pic_file.read((char *) buffer, sizeof(unsigned char) * imgSize);
        cv::Mat row_image(1, (int) imgSize, CV_8U, buffer);
        row_image.row(0).copyTo(training_data.row(i));

        char label = 0;
        label_file.read((char *) &label, sizeof(label));
        label_data.at<uchar>(i, 0) = label;
    }

    return true;
}


int DigitClassifier::classify(cv::Mat img) {
    Mat cloneImg = preprocessImage(img);
    return svm->predict(cloneImg);
    // return knn->findNearest(Mat_<float>(cloneImg), 1);
}

cv::Mat DigitClassifier::preprocessImage(cv::Mat &image) {
//    Mat small = Mat(20,20, CV_8U);
//    cv::resize(image, small, Size(20,20));

    Mat output = Mat(28, 28, CV_8U);
    cv::resize(image, output, Size(28, 28));
    output = output.reshape(0, 1);
    output.convertTo(output, CV_32FC1);
    //std::cout << output.size().width << "x" << output.size().height << std::endl;
    return output;
}
