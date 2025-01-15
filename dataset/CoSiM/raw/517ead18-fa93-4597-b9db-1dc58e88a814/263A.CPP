#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  int row = -1, col = -1;
  for (auto i = 1; i <= 5; i++) {
    for (auto j = 1; j <= 5; j++) {
      int aux;
      cin >> aux;
      if (aux == 1) {
        row = i;
        col = j;
      }
    }
  }
  cout << abs(row - 3) + abs(col - 3);

  return 0;
}