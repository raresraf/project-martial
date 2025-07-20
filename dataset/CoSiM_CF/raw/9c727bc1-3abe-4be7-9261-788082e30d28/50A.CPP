#include <iostream>

using namespace std;

int main() {
  int m, n, s = 0;
  cin >> n >> m;
  if (m % 2 == 0) {
    s = (m / 2) * n;
  } else {
    s = ((m - 1) / 2) * n;
    if (n % 2 == 0) {
      s = s + (n / 2);
    } else {
      s = s + ((n - 1) / 2);
    }
  }
  cout << s;
  return 0;
}
