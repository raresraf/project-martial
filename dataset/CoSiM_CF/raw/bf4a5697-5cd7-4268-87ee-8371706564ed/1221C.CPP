/**
 * I only hope everything will be back to normal
 */
#include <bits/stdc++.h>
#define ll long long

using namespace std;
typedef pair<ll, ll> pii;
typedef vector<ll> vi;
typedef vector<vector<ll>> vii;
typedef vector<vector<pii>> vpii;

int main() {
  int t;
  scanf(" %d", &t);
  while (t--) {
    int c, m, x;
    scanf(" %d %d %d", &c, &m, &x);
    int ans = min(c, min(m, x));
    c -= ans;
    m -= ans;
    x -= ans;
    int pos1, pos2;
    int tempc = c, tempm = m;
    pos1 = min(c >> 1, m);
    tempm -= pos1;
    tempc -= (c >> 1) << 1;
    pos1 += max(min(tempc >> 1, tempm), min(tempm >> 1, tempc));
    tempc = c, tempm = m;
    pos2 = min(m >> 1, c);
    tempm -= (m >> 1) << 1;
    tempc -= pos2;
    pos2 += max(min(tempc >> 1, tempm), min(tempm >> 1, tempc));
    printf("%d\n", ans + max(pos1, pos2));
  }
  return 0;
}