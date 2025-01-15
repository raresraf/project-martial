#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> v;
  int i = 1;
  while (n >= i) {
    v.push_back(i);
    n -= i;
    i++;
  }

  int l = v.size() - 1;
  while (n--) {
    v[l]++;
    l--;
  }
  cout << v.size() << "\n";
  for (auto x : v) {
    cout << x << " ";
  }
}
