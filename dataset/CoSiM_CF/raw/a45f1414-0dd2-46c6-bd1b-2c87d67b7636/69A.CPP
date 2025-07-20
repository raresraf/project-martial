/*====TENSHI29====*/
#include <iostream>
using namespace std;
int main() {
  int n, x = 0, y = 0, z = 0, a, b, c;
  cin >> n;
  while (n--) {
    cin >> a >> b >> c;
    x = x + a;
    y = y + b;
    z = z + c;
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
