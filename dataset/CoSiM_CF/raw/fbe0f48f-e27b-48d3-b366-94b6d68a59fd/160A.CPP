#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <utility>

using namespace std;

int main() {
  int n, k, c, s, i, m, d;
  freopen("in.txt", "r", stdin);
  while (scanf("%d %d", &n, &k) == 2) {
    s = 0;
    for (i = 0; i < n; i++) {
      scanf("%d ", &m);
      c = 0;
      while (m != 0) {
        d = m % 10;
        m /= 10;
        if (d == 7 || d == 4) {
          c++;
        }
      }
      if (c <= k)
        s++;
    }
    printf("%d\n", s);
  }
  return 0;
}
