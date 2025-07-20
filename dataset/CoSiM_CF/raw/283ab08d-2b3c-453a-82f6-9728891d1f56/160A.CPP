#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n, i, k = 0, m = 0, s = 0;
  cin >> n;
  int a[n];
  for (i = 0; i < n; i++)
    cin >> a[i];
  sort(a, a + n);
  for (i = 0; i < n; i++)
    s += a[i];
  for (i = n - 1; i > -1; i--) {
    k = k + a[i];
    s = s - a[i];
    m++;
    if (k > s)
      break;
  }
  cout << m;
  return 0;
}
