// Codeforces 4A - Watermelon.cpp
#include <cstdio>
#include <iostream>

#include <cstring>
#include <sstream>
#include <string>

#include <deque>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

#include <algorithm>
#include <bitset>
#include <functional>
#include <numeric>
#include <utility>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>

#define LL long long
#define MAX 100005
#define MOD 1000000007

#define NL '\n'
using namespace std;

// BEGINTEMPLATE_BY_ZAYED
const double PI = 2 * acos(0.0);
const double EPS = 1e-11;
const int sz = 1e5 + 10;

typedef unsigned long long ull;    // NOTES:"%llu"
typedef long long ll;              // NOTES:"%lld"
typedef long long int64;           // NOTES:int64
typedef unsigned long long uint64; // NOTES:uint64

typedef stack<int> si;
typedef queue<int> qi;
typedef vector<int> vi;
typedef pair<ll, ll> pll;
typedef map<int, int> mii;
typedef pair<int, int> pii;
typedef map<char, int> mci;
typedef map<string, int> msi;
typedef pair<string, int> psi;
typedef map<string, string> mss;

template <class T> inline void checkmin(T &a, T b) {
  if (b < a)
    a = b;
} // NOTES:checkmin(
template <class T> inline void checkmax(T &a, T b) {
  if (b > a)
    a = b;
} // NOTES:checkmax(
template <class T> inline T sqr(T x) { return x * x; } // NOTES:square
int _max(int a, int b) { return (a > b) ? a : b; }
int _min(int a, int b) { return (a < b) ? a : b; }
inline int _pow(int base, int power) {
  int I, result = base;
  if (power == 0)
    return 1;
  for (I = 0; I < power - 1; I++) {
    result = result * base;
  }
  return result;
}
// Translator
bool isUpperCase(char c) { return c >= 'A' && c <= 'Z'; } // NOTES: isUpperCase(
bool isLowerCase(char c) { return c >= 'a' && c <= 'z'; } // NOTES: isLowerCase(
bool isLetter(char c) {
  return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z';
} // NOTES: isLetter(
bool isDigit(char c) { return c >= '0' && c <= '9'; } // NOTES:isDigit(
char toLowerCase(char c) {
  return (isUpperCase(c)) ? (c + 32) : c;
} // NOTES: toLowerCase(
char toUpperCase(char c) {
  return (isLowerCase(c)) ? (c - 32) : c;
} // NOTES: toUpperCase(
int toInt(string s) {
  int r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
} // NOTES:toInt(
double toDouble(string s) {
  double r = 0;
  istringstream sin(s);
  sin >> r;
  return r;
} // NOTES:toDouble(
// ENDTEMPLATE_BY_ZAYED

int main() {
  // std::ios_base::sync_with_stdio(0);
  int cases, caseno = 0, n1, n2, I, K;
  int w;
  scanf("%d", &w);

  if (w == 2) {
    printf("NO\n");
    return 0;
  }

  if (w & 1) {
    printf("NO\n");
    return 0;
  }

  else {
    n1 = 2, n2 = w - 2;
    while (n1 <= n2) {
      if (n1 + n2 == w) {
        printf("YES\n");
        break;
      }
      n1 += 2;
      n2 -= 2;
    }
  }

  return 0;
}
