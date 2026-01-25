#include <iostream>
#include <stdlib.h>

using namespace std;

int main() {
  int mat[5][5], l, k;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      cin >> mat[i][j];
      if (mat[i][j] == 1) {
        l = i;
        k = j;
      }
    }
  int count = 0;
  count = abs(l - 2);
  count = count + abs(k - 2);
  cout << count;
  return 0;
}
