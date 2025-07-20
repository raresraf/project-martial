#include <iostream>

int main() {
  int k, n, w;

  int sum = 0;

  std::cin >> k >> n >> w;

  // calculate sum
  for (int i = 0; i < w; i++) {
    sum += ((i + 1) * k);
  }

  // or using arithmetic progression sum formula => sum = (k + k*w )*w/2

  std::cout << (n >= sum ? 0 : sum - n) << std::endl;

  return 0;
}
