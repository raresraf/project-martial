#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, j, k, l, m, n;
  while (cin >> n) {
    int a[n];
    for (i = 0; i < n; i++) {
      cin >> a[i];
    }
    sort(a, a + n);
    reverse(a, a + n);
    int s = 0, s1 = 0, k = 0, o = 0;
    do {
      k++;
      s = 0, s1 = 0, o = 0;
      for (i = 0; i < k; i++) {
        s += a[i];
      }
      for (j = i; j < n; j++) {
        s1 += a[j];
      }
    } while (s <= s1);
    cout << k << endl;
  }
  return 0;
}
