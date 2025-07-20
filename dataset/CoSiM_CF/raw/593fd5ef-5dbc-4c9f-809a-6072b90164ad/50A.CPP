#include <iostream>

using namespace std;

int main() {
  int m, n, max;
  cin >> m >> n;
  if (n % 2 == 0) {
    max = m * (n / 2);

  } else {
    if (m % 2 == 0) {
      max = m / 2;
    } else {
      max = (m - 1) / 2;
    }
    max = max + m * (n - 1) / 2;
  }
  cout << max;

  return 0;
}
