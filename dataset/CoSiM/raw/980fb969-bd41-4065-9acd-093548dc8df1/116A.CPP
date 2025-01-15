#include <iostream>

int main(int argc, char *argv[]) {
  int n;
  std::cin >> n;

  int min_capacity = 0;
  int number_on_train = 0;
  for (int i = 0; i < n; ++i) {
    int left;
    std::cin >> left;
    number_on_train -= left;

    int entered;
    std::cin >> entered;
    number_on_train += entered;

    min_capacity = std::max(min_capacity, number_on_train);
  }

  std::cout << min_capacity << std::endl;
}
