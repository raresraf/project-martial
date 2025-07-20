#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int x;
  for (int i = 1; i <= 25; i++) {
    cin >> x;
    if (x == 1) {
      if (i % 5 == 0)
        cout << abs(3 - ceil(i / 5.0)) + abs(2) << endl;
      else
        cout << abs(3 - ceil(i / 5.0)) + abs(3 - i % 5) << endl;
    }
  }
}