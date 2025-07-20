#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#define LL long long
using namespace std;
main() {
  char w1[105];
  char w2[105];
  scanf("%s %s", w1, w2);
  int l = strlen(w1);
  int p = 1, i, ans;
  for (i = 0; i < l; i++) {
    w1[i] = tolower(w1[i]);
    w2[i] = tolower((w2[i]));

    if (w1[i] > w2[i]) {
      ans = 1;
      p = 0;
      break;
    }
    if (w1[i] < w2[i]) {
      ans = -1;
      p = 0;
      break;
    }
  }
  if (p == 1)
    ans = 0;

  cout << ans << endl;
  return 0;
}
