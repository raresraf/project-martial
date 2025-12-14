#include <iostream>

using namespace std;
int main() {
  int m, n, num = 0;
  cin >> m >> n;
  while (m >= 2) {
    num = num + n;
    m = m - 2;
  }
  if (m == 1)
    num = num + (n / 2);
  cout << num << endl;
  return 0;
}
