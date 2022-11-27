#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#define MID(x, y) ((x + y) >> 1)
#define L(x) (x << 1)
#define R(x) (x << 1 | 1)
#define FOR(i, s, t) for (int i = (s); i < (t); i++)
#define file_r(x) freopen(x, "r", stdin)
#define file_w(x) freopen(x, "w", stdout)

using namespace std;

int main() {
  int w;
  cin >> w;
  if (w % 2 == 0 && w != 2)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
