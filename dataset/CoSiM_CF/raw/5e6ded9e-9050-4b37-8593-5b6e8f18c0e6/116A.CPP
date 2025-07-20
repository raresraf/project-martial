#include <iostream>

int main() {
  int n;
  std::cin >> n;
  int a[n], b[n], max = 0,
                  j; // j means how many passengers there is after first trip
  for (int i = 0; i < n; i++) { // its easier to calculate from this point
    std::cin >> a[i] >> b[i];
  }

  j = b[0] - a[1] + b[1];

  for (int i = 2; i < n; i++) {
    if (j > max) {
      max = j;
    }
    j = j - a[i] + b[i];
  }

  for (int i = 0; i < n; i++) {
    if (a[i] > max) {
      max = a[i];
    }

    if (b[i] > max) {
      max = b[i];
    }
  }

  std::cout << max;

  return 0;
}