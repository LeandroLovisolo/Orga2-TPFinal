#ifndef __MNIST_LOADER_H__
#define __MNIST_LOADER_H__

#include <fstream>
#include <string>
#include <utility>
#include <vector>

typedef unsigned char uchar;
typedef std::vector<std::pair<std::vector<uchar>, uchar>>  labelled_mnist_data;

class MnistLoader {
 public:
  MnistLoader(const std::string& train_images_path,
              const std::string& train_labels_path,
              const std::string& test_images_path,
              const std::string& test_labels_path);
  inline labelled_mnist_data& train_data() { return train_data_; }
  inline labelled_mnist_data& test_data() { return test_data_; }
  std::string ImageToString(const labelled_mnist_data& data, int index);

 private:
  labelled_mnist_data ReadDataset_(const std::string& images_path,
                                   const std::string& labels_path);
  void ReadHeader_(std::ifstream& file, int& magic_number, int& num_items);
  std::vector<std::vector<uchar>> LoadImagesFile_(const std::string& path);
  std::vector<uchar> LoadLabelsFile_(const std::string& path);

  labelled_mnist_data train_data_;
  labelled_mnist_data test_data_;
};

#endif // __MNIST_LOADER_H__
