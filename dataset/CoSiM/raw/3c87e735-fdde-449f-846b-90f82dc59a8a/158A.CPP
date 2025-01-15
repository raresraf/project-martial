#include <bits/stdc++.h>
using namespace std;
#pragma GCC optimize("-O3")

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  int a, b, n;
  cin >> a >> b >> n;
  a *= 10;
  n -= 1;
  bool found = false;
  for (int i = 0; i < 10; i++) {
    if ((a + i) % b == 0) {
      found = true;
      a += i;
      break;
    }
  }
  if (!found) {
    cout << -1 << "\n";
    return 0;
  }
  cout << a;
  while (n--)
    cout << 0;
  cout << "\n";
  return 0;
}