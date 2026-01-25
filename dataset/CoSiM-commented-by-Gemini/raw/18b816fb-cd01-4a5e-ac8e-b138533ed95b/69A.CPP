#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int subsadd(int x, int y) {
  if (y < 0)
    x -= abs(y);
  else
    x += y;
  return x;
};

int main() {
  int st, y, x, z;
  x = 0;
  y = 0;
  z = 0;
  cin >> st;
  for (int i = 0; i < st; i++) {
    int xx, yy, zz;
    cin >> xx >> yy >> zz;
    x = subsadd(x, xx);
    y = subsadd(y, yy);
    z = subsadd(z, zz);
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}