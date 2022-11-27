// http://codeforces.com/problemset/problem/4/A

#include <bits/stdc++.h>
using namespace std;

int main() {

  long long int x, y;
  cin >> x;
  bool flag = false;

  for (int i = 1; i <= x / 2; i++) {
    if (x % 2 == 0 && (x - i) % 2 == 0) {
      flag = true;
      break;
    }
  }
  if (flag)
    cout << "YES\n";
  else
    cout << "NO\n";
}