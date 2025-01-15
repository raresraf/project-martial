#include <iostream>
#define ll long long
using namespace std;

void solve() {
  ll c, m, x;
  cin >> c >> m >> x;
  ll k = (c + m + x) / 3;
  cout << min(c, min(m, k)) << endl;
}

int main() {
  int q;
  cin >> q;
  while (q--) {
    solve();
  }
}
