#include <algorithm>
#include <bitset>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#define X first
#define Y second
#define MAX(x, y) x = x > (y) ? x : (y)
#define MIN(x, y) x = x > (y) ? (y) : x
#define LOW(x) ((x) & -(x))
#define SQR(x) ((x) * (x))
#define REP(i, n) for (int i = 0; i < (n); ++i)
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
#define rFOR(i, b, a) for (int i = (b); i > (a); --i)
#define CLEAR(x) memset(x, 0, sizeof(x))
#define FILL(x, p) memset(x, p, sizeof(x))
#define COPY(x, y) memcpy(x, y, sizeof(x))
#define ALL(x) (x).begin(), (x).end()

using namespace std;

typedef long long ll;
typedef unsigned int ui;
typedef pair<int, int> PII;
typedef vector<PII> VPII;
typedef vector<int> VI;
typedef vector<string> VS;

const int INF = 0x3fffffff;
const ll _INF = 0x7ffffffffffffffll;
const double EPS = 1e-9;
const double PI = acos(-1.0);

template <class T, class INT> inline T pow(T x, INT y) {
  T res(1);
  for (T tt(x); y; y /= 2) {
    if (y & 1)
      res *= tt;
    tt *= tt;
  }
  return res;
}

template <class T, class INT> inline T pow(T x, INT y, T p) {
  T res(1);
  for (T tt(x); y; y /= 2) {
    if (y & 1)
      res = res * tt % p;
    tt = tt * tt % p;
  }
  return res;
}

int toInt(string s) {
  int x = 0;
  istringstream sin(s);
  sin >> x;
  return x;
}

template <class T> inline string toString(T x) {
  ostringstream sout;
  sout << x;
  return sout.str();
}

template <class INT> inline INT gcd(INT x, INT y) {
  return y ? gcd(y, x % y) : x;
}

template <class INT> inline int calc_bits(INT x) {
  return x ? (x & 1) + calc_bits(x >> 1) : 0;
}

int a1, a2, b1, b2;

inline bool check(const int &a, const int &b) {
  return a - b <= 1 && b <= a + a + 2;
}

int main(int argc, char *argv[]) {
  cin >> a1 >> a2 >> b1 >> b2;
  if (check(a1, b2) || check(a2, b1))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
