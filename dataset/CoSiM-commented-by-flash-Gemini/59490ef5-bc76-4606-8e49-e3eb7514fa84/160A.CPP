#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, sum = 0;
  cin >> n;
  vector<int> v;
  for (int i = 0; i < n; i++) {
    int tmp;
    cin >> tmp;
    sum += tmp;
    v.push_back(tmp);
  }

  sort(v.begin(), v.end());

  for (auto i = v.end(); i != v.begin(); --i) {
  }

  return 0;
}