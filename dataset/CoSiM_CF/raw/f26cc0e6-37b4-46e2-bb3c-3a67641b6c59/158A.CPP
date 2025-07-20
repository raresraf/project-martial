#include <iostream>

int main() {
  int n, k;
  std::cin >> n >> k;
  int count = 0, val;
  for (int i = 0; i < n; ++i) {
    int temp;
    std::cin >> temp;
    if (temp <= 0)
      break;
    else if (i == k - 1) {
      val = temp;
    } else if (i > k - 1) {
      if (temp < val)
        break;
    }
    ++count;
  }
  std::cout << count << '\n';
}
