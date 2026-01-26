#include <iostream>

using namespace std;

int main() {
  int m, n, area;
  int areaOneDominoes = 2;

  cin >> m >> n;
  area = m * n;
  cout << area / areaOneDominoes << endl;

  return 0;
}