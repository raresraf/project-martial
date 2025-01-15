#include <iostream>
#include <stdlib.h>
using namespace std;

int main() {
  int i, j, ct = 0, x;
  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++) {
      cin >> x;
      if (x) {
        ct = abs(i - 2) + abs(j - 2);
        break;
      }
    }
  cout << ct;
  return 0;
}
