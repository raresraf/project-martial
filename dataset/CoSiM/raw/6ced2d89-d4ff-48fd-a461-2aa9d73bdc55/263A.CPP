/*
        SubmissionId	:	2936993
        ContestId	:	263
        Index	:	A
        ProblemName	:	Beautiful Matrix
        ProblemTags	:	['implementation']
        ProgrammingLanguage	:	GNU C++
        Verdict	:	OK
*/

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory.h>
#include <queue>
#include <set>
#include <sstream>
#include <vector>
#pragma comment(linker, "/STACK:16777216")
using namespace std;

typedef long long ll;
typedef long double ld;
typedef double dbl;
typedef float flt;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef pair<double, double> pdd;
template <class F, class T> T convert(F a, int p = -1) {
  stringstream ss;
  if (p >= 0)
    ss << fixed << setprecision(p);
  ss << a;
  T r;
  ss >> r;
  return r;
}
template <class T> void print(T a, int p = -1) {
  if (p >= 0)
    cout << fixed << setprecision(p);
  cout << a;
}
template <class T> T gcd(T a, T b) {
  T r;
  while (b != 0) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}
template <class T> T lcm(T a, T b) { return a / gcd(a, b) * b; }
const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const int dx[] = {-1, 0, +1, 0};
const int dy[] = {0, +1, 0, -1};
const int inf = (int)1e9 + 5;
const ll linf = (ll)1e16 + 5;
const double eps = 1e-9;
const double pi = 3.141592654;
#define vec vector
#define un unsigned
#define pb push_back
#define mp make_pair
#define IT iterator
#define X first
#define Y second
#define debug(x)                                                               \
  cerr << #x << " = " << x << " (line: " << __LINE__ << ")" << endl
#define sqr(a) ((a) * (a))
#define sz(x) ((int)x.size())
#define ms(x, a) memset(x, a, sizeof x)
#define test()                                                                 \
  cerr << "Test!"                                                              \
       << " (line: " << __LINE__ << ")" << endl
#define For(i, a, b) for (typeof(b) i = (a); i <= (b); i++)
#define Ford(i, a, b) for (typeof(a) i = (a); i >= (b); i--)
#define Fit(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); i++)
#define Fitd(i, v)                                                             \
  for (typeof((v).rbegin()) i = (v).rbegin(); i != (v).rend(); i--)
#define all(n) (n).begin(), (n).end()
#define wln(x) cout << x << endl
#define w(x) cout << x
#define r(x, type)                                                             \
  ;                                                                            \
  type x;                                                                      \
  cin >> x;
#define rs(x)                                                                  \
  ;                                                                            \
  string x;                                                                    \
  cin.ignore();                                                                \
  getline(cin, x);
#define rarr(array, i, n)                                                      \
  ;                                                                            \
  for (int i = 0; i <= n; i++)                                                 \
    cin >> array[i];
#define warr(array, i, n)                                                      \
  ;                                                                            \
  for (int i = 0; i <= n; i++)                                                 \
    cout << array[i] << " ";
#define warrd(array, i, n)                                                     \
  ;                                                                            \
  for (int i = n; i >= 0; i--)                                                 \
    cout << array[i] << " ";
#define maxn 100010

int main() {
  ios::sync_with_stdio(false);
  int x, y;
  For(i, 1, 5) For(j, 1, 5) {
    int a;
    cin >> a;
    if (a) {
      x = j;
      y = i;
    }
  }
  cout << abs(x - 3) + abs(y - 3) << endl;
  return 0;
}
