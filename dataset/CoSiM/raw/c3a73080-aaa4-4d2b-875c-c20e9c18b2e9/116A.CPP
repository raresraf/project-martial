#include <bits/stdc++.h>
using namespace std;

int main() {
  int a, b, c, i, j, l, k, o = 0, e = 0;
  cin >> a;
  vector<int> d;
  while (a--) {
    cin >> b;
    cin >> c;
    e = e - b + c;
    d.push_back(e);
  }
  sort(d.begin(), d.end());
  cout << d[d.size() - 1];
}