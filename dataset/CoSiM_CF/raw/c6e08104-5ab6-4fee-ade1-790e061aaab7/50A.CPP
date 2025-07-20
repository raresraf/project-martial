#include <iostream>
using namespace std;
int main() {
  int n, m;
  cin >> m >> n;
  if (m % 2 == 0)
    cout << (n + m / 2);
  else
    cout << (n + m / 2 + n / 2);
  return 0;
}
