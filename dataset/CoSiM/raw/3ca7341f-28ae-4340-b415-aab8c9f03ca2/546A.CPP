#include <bits/stdc++.h>
using namespace std;
int main() {
  long int n, k, w;
  while (cin >> k >> n >> w) {
    long int i, j = 0, l;
    for (i = 1; i <= w; i++) {
      j += (i * k);
    }
    if (j - n > 0) {
      cout << (j - n) << endl;
    } else {
      cout << '0' << endl;
    }
  }
  return 0;
}
