#include <bits/stdc++.h>
using namespace std;

int main() {
  string a, c;
  int i, j, d, e, f;
  cin >> a;
  vector<int> b;
  for (i = 0; i < a.size(); i++) {
    if (i % 2 == 0) {
      b.push_back(a[i]);
    }
  }
  sort(b.begin(), b.end());
  for (i = 0; i < b.size(); i++) {
    if (i != b.size() - 1) {
      c = b[i];
      cout << c;
      cout << "+";
    } else {
      c = b[i];
      cout << c;
    }
  }
}