#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

#include "mnist_loader.h"

using namespace std;

MnistLoader::MnistLoader(const string& train_images_path,
                         const string& train_labels_path,
                         const string& test_images_path,
                         const string& test_labels_path) {
  train_data_ = ReadDataset_(train_images_path, train_labels_path);
  test_data_ = ReadDataset_(test_images_path, test_labels_path);
}

string MnistLoader::ImageToString(const LabelledMistData& data, int index) {
  stringstream ss;
  for(int x = 0; x < 28; x++) {
    for(int y = 27; y >= 0; y--) {
      int offset = 28 * y + x;
      unsigned char c = data[index].first[offset];
      ss << (c < 128 ? "." : "*");
    }
    ss << endl;
  }
  ss << "LABEL: " << to_string(data[index].second) << endl;
  return ss.str();
}

LabelledMistData MnistLoader::ReadDataset_(const std::string& images_path,
                                           const std::string& labels_path) {
  vector<vector<uchar>> images = LoadImagesFile_(images_path);
  vector<uchar> labels = LoadLabelsFile_(labels_path);
  assert(images.size() == labels.size());

  LabelledMistData data;
  for(int i = 0; i < images.size(); i++) {
    data.push_back(make_pair(images[i], labels[i]));
  }
  return data;
}

int reverse_int(int i) {
  unsigned char c1 = i & 255;
  unsigned char c2 = (i >> 8) & 255;
  unsigned char c3 = (i >> 16) & 255;
  unsigned char c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void MnistLoader::ReadHeader_(ifstream& file,
                              int& magic_number,
                              int& num_items) {
  file.read((char *) &magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  file.read((char *) &num_items, sizeof(num_items));
  num_items = reverse_int(num_items);
}

vector<vector<uchar>> MnistLoader::LoadImagesFile_(const string& path) {
  ifstream file(path, ios::binary);
  if(!file.is_open()) {
    throw runtime_error("Cannot open file '" + path + "'.");
  }

  int magic_number, num_images;
  ReadHeader_(file, magic_number, num_images);
  if(magic_number != 2051) throw runtime_error("Invalid MNIST image file.");

  int rows, columns;
  file.read((char *) &rows, sizeof(rows));
  rows = reverse_int(rows);
  file.read((char *) &columns, sizeof(columns));
  columns = reverse_int(columns);

  int image_size = rows * columns;
  vector<vector<uchar>> images(num_images, vector<uchar>(image_size));

  for(int i = 0; i < num_images; i++) {
    file.read((char *) images[i].data(), image_size);
  }

  return images;
}

vector<uchar> MnistLoader::LoadLabelsFile_(const string& path) {
  ifstream file(path, ios::binary);
  if(!file.is_open()) {
    throw runtime_error("Cannot open file '" + path + "'.");
  }

  int magic_number, num_labels;
  ReadHeader_(file, magic_number, num_labels);
  if(magic_number != 2049) throw runtime_error("Invalid MNIST label file.");

  vector<uchar> labels(num_labels);
  file.read((char *) labels.data(), num_labels);
  return labels;
}
