#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int ind[111];
vector<int> v;

int main() {
  int n, k;
  cin >> n >> k;
  for (int i = 0; i < n; ++i) {
    int x;
    cin >> x;
    if (x > 0)
      v.push_back(x);
  }
  ind[0] = 0;
  for (int i = 1; i < v.size(); ++i)
    if (v[i] == v[i - 1])
      ind[i] = ind[i - 1];
    else
      ind[i] = i;
  int ans = 0;
  for (int i = 0; i < v.size(); ++i)
    if (ind[i] < k)
      ++ans;
  cout << ans;
}