#include <iostream>

int main() {
  int i, n, a[1005], b[1005], c[1005], max = 0;
  std::cin >> n;
  for (i = 0; i < n; i++)
    std::cin >> a[i] >> b[i];
  c[0] = b[0];
  for (i = 1; i < n - 1; i++)
    c[i] = c[i - 1] - a[i] + b[i];
  for (i = 0; i < n - 1; i++) {
    if (c[i] > max)
      max = c[i];
  }
  std::cout << max << '\n';
  return 0;
}
