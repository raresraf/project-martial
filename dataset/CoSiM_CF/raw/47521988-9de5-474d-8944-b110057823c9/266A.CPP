// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cctype>
#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  int n, i, j, k, f = 0, s = 0;

  cin >> n;
  char a[n];

  cin >> a;

  for (i = 0; i < n; i++) {
    if (a[i] == a[i + 1]) {
      for (j = i;; j++) {

        if (a[j] != a[i]) {
          i = j;
          break;
        }
        s++;
      }
    }
  }
  cout << --s;
  return 0;
}
