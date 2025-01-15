#include <bits/stdc++.h>
#define ll long long
#define ull unsigned long long
#define MOD 1000000007
using namespace std;

bool solve(vector<vector<vector<int>>> &v, int n, int m) {
  if (m % 2)
    return false;
  for (vector<vector<int>> mat : v)
    if (mat[0][1] == mat[1][0])
      return true;
  return false;
}

int main() {
  int t, n, m, a, b, c, d;
  cin >> t;

  for (int i = 0; i < t; i++) {
    cin >> n >> m;
    vector<vector<vector<int>>> v(n);
    for (int j = 0; j < n; j++) {
      cin >> a >> b >> c >> d;
      v[j] = {{a, b}, {c, d}};
    }
    cout << (solve(v, n, m) ? "YES" : "NO") << endl;
  }
  return 0;
}