#include <iostream>
#include <stdlib.h>
using namespace std;
int main(int argc, char const *argv[]) {
  int ai, aj;
  int k;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> k;
      if (k == 1) {
        ai = i;
        aj = j;
      }
    }
  }
  cout << abs(ai - 2) + abs(aj - 2);
  return 0;
}