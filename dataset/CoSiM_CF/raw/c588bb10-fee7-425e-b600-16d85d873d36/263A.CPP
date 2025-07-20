#include <cstdlib>
#include <iostream>
using namespace std;
int main() {
  int i, j;
  for (int x = 0; x < 5; x++) {
    for (int y = 0; y < 5; y++) {
      int t;
      cin >> t;
      if (t == 1) {
        i = x + 1;
        j = y + 1;
        break;
      }
    }
  }

  // cout << i << j << endl;

  cout << abs(3 - i) + abs(3 - j) << endl;
}
