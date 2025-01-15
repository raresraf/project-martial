#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t, sum = 0, ct = 0;
  vector<int> v;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> t;
    v.push_back(t);
  }
  sort(v.begin(), v.end());
  for (int i = v.size() - 1; i >= 0; i--) {
    ct++;
    sum = sum + v[i];
    if (sum > accumulate(v.begin(), v.end() - ct, 0))
      break;
  }
  cout << ct << endl;
  return 0;
}
