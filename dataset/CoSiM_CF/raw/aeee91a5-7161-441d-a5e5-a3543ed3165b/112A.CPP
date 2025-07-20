#include <iostream>
using namespace std;
int main() {
  string a, b;
  int PointA = 0, PointB = 0;
  cin >> a >> b;

  for (int i = 0; i < a.size(); i++) {
    if ((int)b[i] < 97) {
      b[i] += 32;
    }
    if ((int)a[i] < 97) {
      a[i] += 32;
    }
  }
  if (a > b) {
    cout << "1";
  } else if (b > a) {
    cout << "-1";
  } else
    cout << "0";
  return 0;
}
