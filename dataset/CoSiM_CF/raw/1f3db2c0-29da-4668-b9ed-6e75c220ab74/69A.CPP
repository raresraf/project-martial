#include <iostream>

using namespace std;

int main() {
  int x, y, a = 0, b = 0, c = 0;
  cin >> x;
  for (int i = 0; i < x; i++) {
    cin >> y;
    a += y;
    cin >> y;
    b += y;
    cin >> y;
    c += y;
  }
  if (a == 0 && b == 0 && c == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
