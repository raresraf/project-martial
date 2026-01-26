#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>
#define mset(a, val) memset(a, val, sizeof(a))
#define up(i, s, t) for (int i = (s); i < (t); i += 1)
#define down(i, s, t) for (int i = (s); i > (t); i -= 1)
#define rd1(a) scanf("%d", &a)
#define rd2(a, b) scanf("%d %d", &a, &b)
#define rd3(a, b, c) scanf("%d %d %d", &a, &b, &c)
#define rd4(a, b, c, d) scanf("%d %d %d %d", &a, &b, &c, &d)
#define pii pair<int, int>

typedef long long ll;
using namespace std;
const int MAXINT = 1 << 31 - 1;
const ll MAXLONG = (ll)1 << 63 - 1;

void sti(int a) {
  up(i, 0, a) { printf("|"); }
}

void output(int a, int b, int c) {
  sti(a);
  printf("+");
  sti(b);
  printf("=");
  sti(c);
  cout << endl;
}

bool check_one(int a) {
  if (0 < a)
    return true;
  return false;
}

bool check(int a, int b, int c) {
  if (check_one(a) && check_one(b) && check_one(c)) {
    return true;
  }
  return false;
}

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  string str;
  while (cin >> str) {
    int plus = str.find('+');
    int equa = str.find('=');

    int a = plus;
    int b = equa - plus - 1;
    int c = str.size() - equa - 1;

    if (check(a, b, c) && a + b == c) {
      output(a, b, c);
    } else if (check(a + 1, b, c - 1) && a + b + 1 == c - 1) {
      output(a + 1, b, c - 1);
    } else if (check(a, b + 1, c - 1) && a + b + 1 == c - 1) {
      output(a, b + 1, c - 1);
    } else if (check(a - 1, b, c + 1) && a + b - 1 == c + 1) {
      output(a - 1, b, c + 1);
    } else if (check(a, b - 1, c + 1) && a + b - 1 == c + 1) {
      output(a, b - 1, c + 1);
    } else {
      puts("Impossible");
    }
  }
  return 0;
}