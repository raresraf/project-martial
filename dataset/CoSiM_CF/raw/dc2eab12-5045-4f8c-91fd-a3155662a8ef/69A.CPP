#include <iomanip>
#include <iostream>
#include <math.h>
using namespace std;
int main() {

  int num;
  int x, y, z;
  int vx = 0, vy = 0, vz = 0;
  cin >> num;

  for (int i = 0; i < num; i++) {
    cin >> x >> y >> z;
    vx += x;
    vy += y;
    vz += z;
  }
  if (vx == 0 && vy == 0 && vz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
