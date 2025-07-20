/*http://codeforces.com/problemset/problem/263/A*/

#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int i = 0, j = 0, ipos = 0, jpos = 0, beautifulans = 0;
  int numbers[5][5];
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> numbers[i][j];
      if (numbers[i][j] == 1) {
        ipos = i;
        jpos = j;
      }
    }
  }
  beautifulans = abs(ipos - 2) + abs(jpos - 2);
  cout << beautifulans;
  return 0;
}
