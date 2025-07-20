#include <iostream>
using namespace std;

int main() {
  int m, n, p;
  cin >> m >> n;
  p = m * n;
  if (p % 2 == 0)
    cout << p / 2;
  else
    cout << (p - 1) / 2;
  return 0;
}
