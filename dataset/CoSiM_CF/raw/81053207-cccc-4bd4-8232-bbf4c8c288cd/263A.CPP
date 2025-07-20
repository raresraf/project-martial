#include <bits/stdc++.h>
using namespace std;
#define mp make_pair
#define pb push_back
#define debug(x, y) cout << x << y
#define NL printf("\n")
#define SP printf(" ")
#define mset(x, y) memset(x, y, sizeof(x))
#define mset2d(x, y, m, n) memset(x, y, sizeof(x[0][0] * m * n))

int main() {
  int x, y;
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      int a;
      cin >> a;
      if (a == 1) {
        x = i;
        y = j;
      }
    }
  }
  int res = abs(x - 3) + abs(y - 3);
  cout << res << endl;
  return 0;
}
