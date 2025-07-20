#include <bits/stdc++.h>
using namespace std;

int main() {
  int ara[6][6], i, j, index, key1 = 0, key2 = 0, count = 0, count1 = 0,
                              count2 = 0;
  bool found = false;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> ara[i][j];
    }
  }

  for (i = 1; i < 6; i++) {
    for (j = 1; j < 6; j++) {
      if (ara[i][j] == 1) {
        key1 = i;
        key2 = j;
        break;
        found = true;
      }
    }
    if (found)
      break;
  }

  if (key1 > 3)
    count1 = key1 - 3;
  else
    count1 = 3 - key1;

  if (key2 > 3)
    count2 = key2 - 3;
  else
    count2 = 3 - key2;

  count = count1 + count2;

  cout << count << endl;
}
