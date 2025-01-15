#include <bits/stdc++.h>
#include <iostream>

using namespace std;

void solve() {
  int n;
  vector<string> w;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string cw;
    cin >> cw;
    if (cw.length() > 10) {
      string aux = cw[0] + to_string(cw.length() - 2) + cw[cw.length() - 1];
      w.push_back(aux);
    } else {
      w.push_back(cw);
    }
  }
  for (int j = 0; j < w.size(); j++) {
    cout << w[j] << endl;
  }
}

int main() {
  solve();
  return 0;
}
