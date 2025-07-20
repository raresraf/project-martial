#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  int vc[3];
  int x = 0, y = 0, z = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> vc[0] >> vc[1] >> vc[2];
    x += vc[0];
    y += vc[1];
    z += vc[2];
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}