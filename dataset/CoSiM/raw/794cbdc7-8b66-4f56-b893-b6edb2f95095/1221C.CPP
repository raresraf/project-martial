/*
        O(1) solution
        make c and m equal and then move the rest to x;
        make team with 1c 1m 1x and make the rest into 2c1m or 2m1c
        we can make A 2c1m teams and B 2m1c teams
        so A+B = (c+m)/3
        add up the 1c1m1x team with (c+m)/3 is the answer
*/
#include <bits/stdc++.h>

using namespace std;

int main() {
  int q, c, m, x;
  cin >> q;
  for (int i = 0; i < q; i++) {
    int ans = 0;
    cin >> c >> m >> x;
    if (c > m) {
      x += c - m;
      c = m;
    } else {
      x += m - c;
      m = c;
    }
    ans = min(c, min(m, x));
    c -= ans;
    m -= ans;
    ans += (c + m) / 3;
    cout << ans << endl;
  }

  return 0;
}