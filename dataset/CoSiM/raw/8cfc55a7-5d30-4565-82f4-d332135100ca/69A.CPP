#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x = 0, y = 0, z = 0;
  while (n > 0) {
    int t1, t2, t3;
    cin >> t1 >> t2 >> t3;
    x = x + t1;
    y = y + t2;
    z = z + t3;
    n--;
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";
}