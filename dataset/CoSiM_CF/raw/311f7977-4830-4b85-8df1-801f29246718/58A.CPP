/* Date: 10.01.16
Problem: 58A - Chat room
*/

#include <cstdio>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

int compare(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

int main() {
  string c;
  char b[] = {'h', 'e', 'l', 'l', 'o'};
  cin >> c;

  int i, p = 0, x = 0, j, l = c.length();

  for (i = 0; i < 5; i++) {

    for (j = x; j < l; j++) {

      if (b[i] == c[j]) {
        x = j + 1;
        p = 1;
        break;
      } else
        p = 0;
    }
    if (j == l) {
      break;
    }
  }
  if (j == l) {
    cout << "NO\n";
  } else if (i == 5 && p == 1) {
    cout << "YES\n";
  }

  return 0;
}
