/*=====TENSHI29=====*/
#include <cmath>
#include <cstdio>
#include <iostream>
using namespace std;
int main() {
  int mat[5][5], ans, I, J;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> mat[i][j];
      if (mat[i][j] == 1) {
        I = i;
        J = j;
        // goto print;
      }
    }
  }
  // print:
  ans = abs(I - 2) + abs(J - 2);
  cout << ans << endl;
  return 0;
}
