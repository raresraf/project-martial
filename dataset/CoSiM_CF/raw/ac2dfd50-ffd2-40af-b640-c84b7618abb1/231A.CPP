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

  int n, cnt, sum = 0;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    cnt = 0;
    for (int j = 1; j <= 3; j++) {
      int a;
      cin >> a;
      if (a == 1)
        cnt++;
    }
    if (cnt >= 2)
      sum++;
  }
  cout << sum << endl;
  return 0;
}
