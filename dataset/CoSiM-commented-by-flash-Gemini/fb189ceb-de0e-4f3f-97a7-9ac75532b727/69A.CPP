#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
main() {
  int n;
  cin >> n;
  long long sumx = 0;
  long long sumy = 0;
  long long sumz = 0;

  int x, y, z;
  for (int i = 0; i < n; i++) {
    cin >> x;
    sumx += x;
    cin >> y;
    sumy += y;
    cin >> z;
    sumz = sumz + z;
  }

  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
}
