/// From dust I came and to dust I shall return
#include <bits/stdc++.h>
using namespace std;
int main() {
  int m, n, x, sum = 0, i;
  cin >> n;
  m = n * 3;
  for (i = 0; i < m; i++) {
    cin >> x;
    sum += x;
  }
  if (sum == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
