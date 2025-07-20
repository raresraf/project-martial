#include "limits.h"
#include <iostream>

int main() {
  int n, k;
  int kth, now;
  kth = -32767;

  std::cin >> n >> k;

  int count = 0;
  for (int i = 0; i < n; i++) {
    std::cin >> now;
    if (i == k - 1) {
      kth = now;
    }
    // stop early if now is 0 or smaller than kth value
    if (now == 0 || now < kth) {
      break;
    }
    count++;
  }
  std::cout << count << std::endl;
  return 0;
}