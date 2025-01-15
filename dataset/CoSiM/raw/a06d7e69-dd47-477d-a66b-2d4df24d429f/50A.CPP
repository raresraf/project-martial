#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, m;
  cin >> n >> m;
  if (n == 1 && m == 1) {
    cout << 0;
  } else {
    cout << (m * n) / 2;
  }
  return 0;
}
