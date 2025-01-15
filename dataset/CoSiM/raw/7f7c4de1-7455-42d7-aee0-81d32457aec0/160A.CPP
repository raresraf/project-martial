#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  int n;
  std::cin >> n;
  std::vector<int> values(n);
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    std::cin >> values[i];
    sum += values[i];
  }
  std::sort(values.begin(), values.end());
  int sum2 = 0;
  int j;
  for (int i = n - 1; i >= 0; --i) {
    sum2 += values[i];
    if (sum2 > sum - sum2) {
      std::cout << n - i << '\n';
      return 0;
    }
  }
}
