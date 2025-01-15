#include <bits/stdc++.h>
using namespace std;

int main() {
  int k;
  cin >> k;

  int ans = 0;
  int comp = 0;
  for (int i = 0; i < k; ++i) {
    int n, m;
    cin >> n >> m;
    comp += (m - n);
    /* cout <<  "max" << "(" << ans <<  "," << ans+comp << ")" << '\n'; */
    ans = max(ans, comp);
  }
  cout << ans << '\n';
}
