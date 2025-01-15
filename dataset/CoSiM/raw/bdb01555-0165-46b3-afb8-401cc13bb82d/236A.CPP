#include <algorithm>
#include <assert.h>
#include <iostream>
#include <limits.h>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <utility>
#include <vector>

using namespace std;

#define rep(i, n) for (__typeof(n) i = 0; (i) < (n); ++i)
#define fu(i, a, b) for (__typeof(a) i = (a); (i) <= (b); ++i)
#define fd(i, a, b) for (__typeof(a) i = (a); (i) >= (b); --i)
#define tr(v, it) for (__typeof(v.begin()) it = v.begin(); it != v.end(); ++it)

typedef long long ll;
typedef unsigned long long ull;
typedef vector<int> vi;
typedef pair<int, int> pii;
typedef long double ld;

#define pb push_back
#define mp make_pair
#define linf LONG_LONG_MAX
#define inf INT_MAX
#define X first
#define Y second
#define fill(a, val) memset(a, val, sizeof(a))
#define size(v) (int)v.size()
#define gi(n) scanf("%d", &n)
#define all(v) v.begin(), v.end()
#define debug(format, args...)                                                 \
  {                                                                            \
    cerr << #args << " -> ";                                                   \
    fprintf(stderr, format, args);                                             \
    cerr << "\n";                                                              \
  }

int main() {
  int k, a, b, v;
  cin >> k >> a >> b >> v;

  int countBox = 0, countDiv = 0, countSec = 0;

  while (a > 0) {
    countSec = 1;
    while ((a - v) >= 0 && (countSec <= k && countDiv <= b)) {
      a = a - v;
      // cout << a << endl;
      countDiv++;
      // if(countDiv == b + 1) break;
      countSec++;
    }

    countBox++;
    if ((a - v) < 0) {
      break;
    }
    // cout << "count" << countBox << endl;
    if (countDiv == b || countDiv == b + 1) {
      break;
    }
  }

  // cout << "a" << a << endl;

  if (a > 0) {
    countBox += a / v;
    if (a % v != 0) {
      countBox += 1;
    }
    // cout << countBox << "asdk;lad" << endl;
  }

  cout << countBox << endl;

  return 0;
}