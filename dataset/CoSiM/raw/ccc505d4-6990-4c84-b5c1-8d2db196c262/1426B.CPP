#include <bits/stdc++.h>
using namespace std;
#define optimizar_io                                                           \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);

typedef long long int ll;

int main() {
  optimizar_io int tc, n, m;

  cin >> tc;
  while (tc--) {
    cin >> n >> m;
    bool possible = false;
    for (int i = 0; i < n; i++) {
      vector<vector<int>> v(2, vector<int>(2, 0));
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++)
          cin >> v[j][k];
      }
      possible |= (v[0][1] == v[1][0]);
    }
    if (m & 1) {
      cout << "NO\n";
      continue;
    }
    cout << (possible ? "YES\n" : "NO\n");
  }
  return 0;
}
