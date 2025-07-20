#include <iostream>

using namespace std;

int main() {
  int totVectors, x = 0, y = 0, z = 0, input;

  cin >> totVectors;

  for (int i = 0; i < totVectors * 3; i++) {
    if (i % 3 == 0) {
      cin >> input;
      x += input;
    }

    else if (i % 3 == 1) {
      cin >> input;
      y += input;
    }

    else if (i % 3 == 2) {
      cin >> input;
      z += input;
    }
  }

  if ((x == 0) && (y == 0) && (z == 0)) {
    cout << "YES" << endl;
  }

  else {
    cout << "NO" << endl;
  }

  // cout << x << endl << y << endl << z;

  return 0;
}
