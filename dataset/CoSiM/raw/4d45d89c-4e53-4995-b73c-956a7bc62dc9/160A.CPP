// 28.01.19 @odnaks
#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int a[n];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  sort(&a[0], &a[n]);
  reverse(&a[0], &a[n]);
  int s1;
  int s2;
  for (int m = 1; m <= n; m++) {
    s1 = 0;
    s2 = 0;
    for (int i = 0; i < m; i++)
      s1 += a[i];
    for (int i = m; i < n; i++)
      s2 += a[i];
    if (s1 > s2) {
      cout << m;
      break;
    }
  }
  return 0;
}