/*
Link: https://codeforces.com/problemset/problem/263/A
@uthor: chavodon
*/

#include <cstdlib>
#include <iostream>

using namespace std;

int main() {
  int matrix[5][5];
  int r, trans, n1, n2, n3, n4, n5, rowpos, colpos;
  for (r = 0; r <= 4; r++) {
    cin >> n1 >> n2 >> n3 >> n4 >> n5;
    matrix[r][0] = n1;
    matrix[r][1] = n2;
    matrix[r][2] = n3;
    matrix[r][3] = n4;
    matrix[r][4] = n5;

    if (n1 == 1) {
      rowpos = r;
      colpos = 0;
    } else if (n2 == 1) {
      rowpos = r;
      colpos = 1;
    } else if (n3 == 1) {
      rowpos = r;
      colpos = 2;
    } else if (n4 == 1) {
      rowpos = r;
      colpos = 3;
    } else if (n5 == 1) {
      rowpos = r;
      colpos = 4;
    }
  }
  cout << abs(2 - rowpos) + abs(2 - colpos);
  return 0;
}