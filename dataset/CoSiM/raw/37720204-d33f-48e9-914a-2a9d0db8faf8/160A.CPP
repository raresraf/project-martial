// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  int n, k, s = 0, dr = 0, a[100], i, f1, f2, j, temp, c = 0;
  long int m;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    s = s + a[i];
  }

  for (i = n - 2; i >= 0; i--) {
    for (j = 0; j <= i; j++) {
      if (a[j] < a[j + 1]) {
        temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
      }
    }
  }

  for (i = 0; i < n; i++) {
    c = c + a[i];
    if (c > (s / 2)) {
      break;
    }
  }
  i++;
  cout << i;
}
