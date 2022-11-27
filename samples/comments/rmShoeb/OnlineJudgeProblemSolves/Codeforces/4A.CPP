#include <iostream>
using namespace std;

int main(void) {
  int i, x, a, b;

  cin >> x;

  if (x >= 4 && x % 2 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
