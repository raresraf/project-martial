#include <iostream>
using namespace std;
int main() {

  int n;
  int xcount, ycount, zcount;
  xcount = ycount = zcount = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {
    int x, y, z;
    cin >> x;
    cin >> y;
    cin >> z;
    xcount += x;
    ycount += y;
    zcount += z;
  }

  if (xcount == 0 && ycount == 0 && zcount == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
