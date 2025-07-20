#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n, k, c(0);
  cin >> n >> k;
  vector<int> v(n);

  for (int i = 1; i <= n; i++) {
    cin >> v[i];
  }

  for (int i = 1; i <= n; i++) {
    if (v[i] != NULL && v[i] >= v[k])
      c++;
  }
  cout << c << "\n";
}
