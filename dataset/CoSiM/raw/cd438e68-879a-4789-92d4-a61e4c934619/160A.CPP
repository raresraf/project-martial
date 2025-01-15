#include <iostream>
using namespace std;

int main() {
  int a[100], n, i, j, ct = 0, s = 0, r = 0, k, t;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    s += a[i];
  }
  for (i = 0; i < n - 1; i++)
    for (j = 0; j < n - 1 - i; j++)
      if (a[j] > a[j + 1]) {
        t = a[j];
        a[j] = a[j + 1];
        a[j + 1] = t;
      }
  i = n - 1;
  while (r <= s) {
    r += a[i];
    s -= a[i];
    i--;
    ct++;
  }
  cout << ct;
  return 0;
}
