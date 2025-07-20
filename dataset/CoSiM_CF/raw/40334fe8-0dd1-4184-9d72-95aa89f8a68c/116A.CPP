#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  int mc = 0, cc = 0;
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    cc = cc - a + b;
    if (cc > mc)
      mc = cc;
  }
  cout << mc;
}