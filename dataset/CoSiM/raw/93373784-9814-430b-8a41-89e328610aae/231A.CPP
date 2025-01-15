#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, ct = 0;
  cin >> t;
  while (t--) {
    int a, b, c;
    cin >> a >> b >> c;
    if (a + b + c >= 2)
      ct++;
  }
  cout << ct << endl;
  return 0;
}
