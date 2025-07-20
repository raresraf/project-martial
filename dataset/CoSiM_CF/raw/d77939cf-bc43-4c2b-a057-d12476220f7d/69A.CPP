#include <algorithm>
#include <array>
#include <bitset>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

using namespace std;

#define _USE_MATH_DEFINES
#define abs(a) (((a) > 0) ? (a) : (-(a)))
#define M_PI 3.14159265358979323846 /* pi */
#define LL long long int
#define vt vector

void solve() {
  int n, x, y, z, xs = 0, xy = 0, xz = 0;
  cin >> n;
  while (n--) {
    cin >> x >> y >> z;
    xs += x;
    xy += y;
    xz += z;
  }
  if (xs == 0 && xy == 0 && xz == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}

int main() { solve(); }
