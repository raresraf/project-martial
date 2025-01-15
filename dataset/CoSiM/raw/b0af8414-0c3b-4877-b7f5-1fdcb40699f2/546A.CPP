#include <iostream>
using namespace std;
int main() {
  int a, b, c, d = 0;
  cin >> a >> b >> c;
  for (int i = 1; i <= c; i++) {
    d += a * i;
  }
  if (d > b) {
    cout << d - b;
  } else
    cout << "0";
}
