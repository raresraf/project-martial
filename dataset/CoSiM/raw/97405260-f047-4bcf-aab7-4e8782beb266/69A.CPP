#include <bits/stdc++.h>
#include <iostream>
#define ll long long
#define MOD 1000000007
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  ll curX = 0, curY = 0, curZ = 0;
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int x, y, z;
    cin >> x >> y >> z;
    curX += x, curY += y, curZ += z;
  }

  if (curX == 0 && curY == 0 && curZ == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}