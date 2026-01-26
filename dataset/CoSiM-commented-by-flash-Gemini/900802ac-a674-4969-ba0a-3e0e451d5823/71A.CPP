#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#define vc vector<int>
#define gcd(x, y) __gcd(x, y)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define MAX 1000001
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  int tc;
  string s;
  cin >> tc;
  while (tc--) {
    cin >> s;
    int len = s.size();
    if (len <= 10) {
      cout << s << endl;
      continue;

    } else {
      printf("%c%d%c\n", s[0], len - 2, s[len - 1]);
    }
  }
  return 0;
}
