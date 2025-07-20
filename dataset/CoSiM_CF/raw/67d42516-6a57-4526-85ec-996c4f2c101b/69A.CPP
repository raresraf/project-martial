// Written By Deepesh Nimma
// Problem: 69A Codeforces
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define space " "
#define end1 "\n"
#define ar array

int main() {
  int n;
  int xsum = 0, ysum = 0, zsum = 0;
  cin >> n;

  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;

    xsum += x;
    ysum += y;
    zsum += z;
  }

  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES" << end1;
    return 0;
  } else {
    cout << "NO" << end1;
  }

  return 0;
}
