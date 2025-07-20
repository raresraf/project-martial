// https://codeforces.com/problemset/problem/1343/C
// SOLUTION BY DN54321

#include <algorithm>  //sort
#include <ctype.h>    // toupper, tolower, islower, isupper
#include <functional> // greater, less
#include <iostream>   // printf/scanf
#include <numeric>    //gcd, lcm (c++17)
#include <set>        // unsorted set, multiset, set
#include <string.h>   // strlen
#include <vector>     // vector
#define p1 first
#define p2 second
#define pii pair<int, int>
#define MXINT (1 << 30)

typedef long long ll;
using namespace std;

int main() {
  int t, n, x, c;
  scanf("%d", &t);
  for (int tc = 0; tc < t; ++tc) {
    scanf("%d%d", &n, &c);
    ll r = 0;
    for (int i = 1; i < n; ++i) {
      scanf("%d", &x);
      if (1LL * x * c < 0LL) {
        r += c;
        c = x;
      } else if (x > c)
        c = x;
    }
    r += c;
    printf("%I64d\n", r);
  }
  return 0;
}