#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int k, r = 0;

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> k;
      if (k) {
        if (j > 2)
          r += j - 2;
        else if (j < 2)
          r += 2 - j;
        if (i > 2)
          r += i - 2;
        else if (i < 2)
          r += 2 - i;
      }
    }
  }

  cout << r << endl;

  return 0;
}
