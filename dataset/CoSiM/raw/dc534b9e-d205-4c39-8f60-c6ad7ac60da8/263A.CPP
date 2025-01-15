#include <iostream>
using namespace std;
int main() {
  int i, j, t, ai, aj;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> t;
      if (t == 1) {
        ai = i;
        aj = j;
      }
    }
  }
  cout << abs(ai - 3) + abs(aj - 3) << endl;
  return 0;
}
