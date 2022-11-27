#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  if (n % 2 != 0 || (n / 2) % 2 != 0) {
    for (int i = 3; i < n; i++) {
      if (n % i == 0 && (n / i) % 2 == 0) {
        cout << "YES" << endl;
        return 0;
      }
    }
    cout << "NO" << endl;
  } else
    cout << "YES" << endl;

  return 0;
}
// http://codeforces.com/problemset/problem/4/A