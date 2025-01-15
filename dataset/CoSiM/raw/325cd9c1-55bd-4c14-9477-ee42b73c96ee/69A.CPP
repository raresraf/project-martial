#include <iostream>
using namespace std;
int main() {
  int x = 0, y = 0, z = 0;
  int n;
  cin >> n;
  while (n--) {
    int i, j, k;
    cin >> i >> j >> k;
    x += i;
    y += j;
    z += k;
  }
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";
}