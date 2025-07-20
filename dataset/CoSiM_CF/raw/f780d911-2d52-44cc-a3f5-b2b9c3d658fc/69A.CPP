#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int cs, x, y, z, xt, yt, zt;
  while (cin >> cs) {
    xt = yt = zt = 0;
    for (int i = 0; i < cs; i++) {
      cin >> x >> y >> z;
      xt += x;
      yt += y;
      zt += z;
    }
    if (xt == 0 && yt == 0 && zt == 0) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}
