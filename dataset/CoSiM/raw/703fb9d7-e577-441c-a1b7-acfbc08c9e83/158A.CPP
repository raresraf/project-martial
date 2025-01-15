#include <iostream>

int main() {
  int n = 0, k = 0;
  int *a = nullptr;
  int passingScore = 0;
  int counter = 0;

  std::cin >> n >> k;

  a = new int[n];

  for (int i = 0; i < n; i++) {
    std::cin >> a[i];

    // std::cout << a[i] << std::endl;
  }

  passingScore = a[k - 1];

  for (int i = 0; i < n; i++) {
    if (a[i] < passingScore || a[i] <= 0) {
      break;
    } else {
      counter++;
    }
  }

  std::cout << counter << std::endl;

  return 0;
}
