// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  int n, i, j, k, a, b, c, cnt = 0;
  ;

  cin >> n;

  for (i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if (a == 1 && (b == 1 || c == 1)) {
      cnt++;
    } else if (b == 1 && (a == 1 || c == 1)) {
      cnt++;
    } else if (c == 1 && (b == 1 || a == 1)) {
      cnt++;
    }
    a = 0;
    b = 0;
    c = 0;
  }
  cout << cnt;

  return 0;
}
