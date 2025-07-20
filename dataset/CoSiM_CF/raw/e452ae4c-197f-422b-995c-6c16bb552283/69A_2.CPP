#include <iostream>

using namespace std;

int main() {
  int x, y, z, n, sumOfx = 0, sumOfy = 0, sumOfz = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    sumOfx += x;
    sumOfy += y;
    sumOfz += z;
  }
  if (!sumOfx && !sumOfy && !sumOfz)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
