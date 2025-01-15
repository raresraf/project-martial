#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n;
  cin >> n;
  int xvec, yvec, zvec;
  int x = 0, y = 0, z = 0;

  while (n--) {
    cin >> xvec >> yvec >> zvec;
    x += xvec;
    y += yvec;
    z += zvec;
  }

  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
