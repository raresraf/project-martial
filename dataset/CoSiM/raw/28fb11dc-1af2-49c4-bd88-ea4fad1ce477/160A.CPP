// Abhishek Gupta | TCET,Mumbai | BE.IT

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, i, x;
  cin >> n;
  int s = 0, s1 = 0, c = 0;
  int a[n];
  for (i = 0; i < n; i++) {
    cin >> a[i];
    s += a[i];
  }
  sort(a, a + n, greater<int>());
  s /= 2;
  for (i = 0; i < n; i++) {
    s1 += a[i];
    c++;
    if (s1 > s) {
      break;
    }
  }
  cout << c;
  return 0;
}
