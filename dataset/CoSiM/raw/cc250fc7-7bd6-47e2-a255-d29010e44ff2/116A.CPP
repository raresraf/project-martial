#include <iostream>

using namespace std;

int main() {
  int n, c = 0;

  cin >> n;

  int a[n], b[n], d[n];

  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }

  d[0] = b[0] - a[0];

  for (int i = 1; i < n; i++) {
    d[i] = d[i - 1] - a[i] + b[i];
  }

  for (int i = 0; i < n; i++) {
    if (d[i] >= c) {
      c = d[i];
    }
  }

  cout << c << endl;

  return 0;
}
