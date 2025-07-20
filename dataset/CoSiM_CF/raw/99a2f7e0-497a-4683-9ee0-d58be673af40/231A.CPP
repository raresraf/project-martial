#include <iostream>

int main() {
  int n, nb_pb = 0, acc, temp;
  std::cin >> n;
  for (int i = 0; i < n; i++) {
    acc = 0;
    for (int j = 0; j < 3; j++) {
      std::cin >> temp;
      acc += temp;
    }
    if (acc >= 2)
      nb_pb++;
  }
  std::cout << nb_pb;
}
