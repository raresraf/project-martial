#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n, m = 0;
  cin >> n;
  int a, b, c;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if (a + b + c >= 2) {
      m += 1;
    }
  }
  cout << m << endl;

  return 0;
}
