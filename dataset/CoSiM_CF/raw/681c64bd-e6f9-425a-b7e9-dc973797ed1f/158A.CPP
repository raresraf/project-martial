#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, k;
  cin >> n >> k;

  vector<int> v;

  for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    v.push_back(x);
  }
  int r = k - 1;
  while (r < n - 1 && v[r + 1] == v[k - 1])
    r++;

  while (r >= 0 && v[r] == 0)
    r--;

  cout << r + 1 << endl;

  return 0;
}
