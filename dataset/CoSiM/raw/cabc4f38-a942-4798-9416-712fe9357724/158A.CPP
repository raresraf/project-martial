#include <iostream>

using namespace std;

int main() {
  int n, k, count = 0;
  std::cin >> n >> k;
  int a[n];
  for (int i = 1; i <= n; i++) {
    std::cin >> a[i];
  }
  for (int j = 1; j <= n; j++) {
    if (a[j] > 0 && a[j] >= a[k])
      count++;
  }
  std::cout << count;
  return 0;
}
