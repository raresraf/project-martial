#include <iostream>

int main() {
  unsigned n{}, k{}, count{}, temp{};
  signed short temp1 = -1;
  do {
    std::cin >> n >> k;
  } while ((n < k) || (n < 1 || n > 50) || (k < 1 || k > 50));
  for (auto i = 0; i < n; i++) {
    std::cin >> temp;
    if (i == k - 1)
      temp1 = temp;
    if (temp1 > 0) {
      if (temp1 == temp)
        count++;
      continue;
    }
    if (temp > 0)
      count++;
  }
  std::cout << count;

  return 0;
}
