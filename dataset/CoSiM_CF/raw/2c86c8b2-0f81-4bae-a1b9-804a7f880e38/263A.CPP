#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main() {
  int input[5][5];
  int min;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      scanf("%d", &input[i][j]);
      if (input[i][j] == 1) {
        min = abs(2 - i) + abs(j - 2);
      }
    }
  }
  cout << min;
}
