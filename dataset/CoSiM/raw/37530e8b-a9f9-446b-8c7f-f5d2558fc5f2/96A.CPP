// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  int n, i, j, k;
  char a[100];
  cin >> a;
  if (strlen(a) > 7) {
    for (i = 0; i < (strlen(a) - 6); i++) {
      if (a[i] == a[i + 1] && a[i] == a[i + 2] && a[i] == a[i + 3] &&
          a[i] == a[i + 4] && a[i] == a[i + 5] && a[i] == a[i + 6]) {
        cout << "YES";
        return 0;
      }
    }
  }
  cout << "NO";

  return 0;
}
