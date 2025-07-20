#include <bits/stdc++.h>
using namespace std;

int main() {
  string a, c;
  int b, i, j, d, e, f = 0;
  cin >> b;
  cin >> a;
  vector<int> k;
  for (i = 0; i < b; i++) {
    k.push_back(a[i]);
  }
  for (i = 0; i < b; i++) {
    if (k[i] == k[i + 1]) {
      f = f + 1;
    }
  }
  cout << f;
}