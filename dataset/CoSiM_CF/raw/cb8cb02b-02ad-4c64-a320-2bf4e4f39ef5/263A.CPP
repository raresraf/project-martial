#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int x, pi, pj;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> x;
      if (x == 1) {
        pi = i;
        pj = j;
      }
    }
  }
  x = abs(2 - pi) + abs(2 - pj);
  cout << x;
  return 0;
}
