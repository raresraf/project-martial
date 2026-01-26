#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int i = 0, j = 0, a0, a1, a2, a3, a4;
  long int row, col;
  string s[5];
  char tem[10];
  while (i < 5) {
    j = 0;
    s[i] = "";
    while (j < 5) {
      cin >> a0;
      sprintf(tem, "%d", a0);
      s[i] = s[i] + tem + " ";
      if (a0 == 1) {
        row = i;
        col = j;
      }
      j++;
    }
    i++;
  }
  cout << fabs(row - 2) + fabs(col - 2);
  return 0;
}
