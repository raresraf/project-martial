#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, b, i, j, l = 0, o, k;
  cin >> n;
  vector<int> a;
  k = n;
  while (k--) {
    cin >> b;
    a.push_back(b);
  }
  sort(a.begin(), a.end());
  reverse(a.begin(), a.end());
  for (i = 0; i < n; i++) {
    l = l + a[i];
    o = 0;
    for (j = i + 1; j < n; j++) {
      o = o + a[j];
    }
    if (l > o) {
      cout << i + 1;
      break;
    }
  }
  return 0;
}
