#include <bits/stdc++.h>
using namespace std;

int main() {
  int c = 0, t, h, n, k;
  cin >> n >> k;
  if (n < k)
    cout << 0;
  else {
    cin >> h;
    t = h;
    for (int i = 1; i <= k; ++i) {
      if (t > 0 && t <= h) {
        c++;
      } else
        break;
      cin >> t;
      if (t < h && c < k)
        h = t;
    }
    // cout<<h<<c;
    while (t > 0 && t == h && c < n) {
      c++;
      cin >> t;
    }
  }
  cout << c;
}