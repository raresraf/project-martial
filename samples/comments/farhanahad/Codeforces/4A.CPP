#include <iostream>
using namespace std;
int main() {
  long long int x, n, y, z;

  cin >> x >> y >> n;
  z = ((n * (n + 1) * x) / 2);
  if (z - y < 0)
    cout << "0";
  else
    cout << z - y;
}
