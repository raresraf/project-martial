#include <cmath>
#include <iostream>

using namespace std;

int main() {

  int x = 0;
  int i;
  int j;

  for (i = 1; i <= 5; i++) {

    for (j = 1; j <= 5; j++) {

      cin >> x;

      if (x == 1) {

        cout << abs(i - 3) + abs(j - 3) << endl;
      }
    }
  }

  return 0;
}
