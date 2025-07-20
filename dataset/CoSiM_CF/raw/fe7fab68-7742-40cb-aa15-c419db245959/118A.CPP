// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  int n, i, j = -1, k;
  char a[110], b[210] = {0}, t;
  cin >> a;

  for (i = 0; i < strlen(a); i++) {
    if (a[i] == 'a' || a[i] == 'e' || a[i] == 'i' || a[i] == 'o' ||
        a[i] == 'u' || a[i] == 'y' || a[i] == 'A' || a[i] == 'E' ||
        a[i] == 'I' || a[i] == 'O' || a[i] == 'U' || a[i] == 'Y') {
      continue;
    } else if (a[i] >= 'A' && a[i] <= 'Z') {
      j++;
      b[j] = '.';
      j++;
      t = a[i];
      b[j] = tolower(t);

    } else if (a[i] >= 'a' && a[i] <= 'z') {
      j++;
      b[j] = '.';
      j++;
      t = a[i];
      b[j] = t;
    }
  }
  for (i = 0; i < strlen(b); i++) {
    cout << b[i];
  }
  return 0;
}
