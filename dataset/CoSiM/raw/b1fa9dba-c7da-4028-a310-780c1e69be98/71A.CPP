// http://codeforces.com/problemset/problem/71/A

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <queue>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#ifdef TEST
#include "tools.h"
#endif
using namespace std;

int main() {

#ifdef TEST
  freopen("test.in", "r", stdin);
  freopen("test.out", "w", stdout);
#endif

  string s = "";
  int n;
  cin >> n;
  for (int i = 0; i < n; ++i) {

    cin >> s;
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }

  return 0;
}