#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string w;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> w;
    if (w.length() > 10) {
      cout << w.at(0) << w.length() - 2 << w.back() << endl;
    } else {
      cout << w << endl;
    }
  }
  return 0;
}
