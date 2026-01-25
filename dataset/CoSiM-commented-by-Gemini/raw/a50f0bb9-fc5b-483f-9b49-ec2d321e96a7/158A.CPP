#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, e, i, a, c = 0;
  cin >> n;
  cin >> k;
  vector<int> b;
  for (i = 0; i < n; i++) {
    cin >> a;
    b.push_back(a);
  }
  e = b[k - 1];
  for (i = 0; i < n; i++) {
    if (e <= b[i] && b[i] > 0) {
      c = c + 1;
    }
  }
  cout << c;
}