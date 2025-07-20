// Codeforces 281A - Word Capitalization.cpp
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

int caseno = 1;
#define NL '\n'
#define FOR(I, J, K) for (I = J; I < K; I++)
#define REV(I, J, K) for (I = J; I > K; I--)
#define sf1(x) scanf("%d", &x)
#define sf scanf
#define pf1(x) printf("%d", x)
#define pf printf
#define PC() printf("Case %d: ", caseno++)
#define READ() freopen("in.txt", "r", stdin)
#define WRITE() freopen("out.txt", "w", stdout)
using namespace std;

// BEGINTEMPLATE_BY_ZAYED
typedef unsigned long long ULL;    // NOTES:"%llu"
typedef unsigned long long uint64; // NOTES:uint64
typedef long long LL;              // NOTES:"%lld"
typedef long long int64;           // NOTES:int64

template <class T> inline void checkmax(T &a, T b) {
  if (b > a)
    a = b;
} // NOTES:checkmax(
template <class T> inline void checkmin(T &a, T b) {
  if (b < a)
    a = b;
} // NOTES:checkmin(
template <class T> inline T MAX(T a, T b) { return (a > b) ? a : b; }
template <class T> inline T MIN(T a, T b) { return (a < b) ? a : b; }
template <class T> inline T sqr(T x) { return x * x; } // NOTES:square
template <class T> inline T fABS(T a) { return a < 0 ? a * (-1) : a; }
template <class T> inline void SWAP(T &a, T &b) {
  T t = a;
  a = b;
  b = t;
}
inline LL POW(LL base, LL power) {
  LL I, res = base;
  if (power == 0)
    return 1;
  for (I = 0; I < power - 1; I++)
    res *= base;
  return res;
}
// Translator
bool isUpperCase(char c) { return c >= 'A' && c <= 'Z'; } // NOTES: isUpperCase(
bool isLowerCase(char c) { return c >= 'a' && c <= 'z'; } // NOTES: isLowerCase(
bool isVowel(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
          c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
} // NOTES: isVowel(
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
string toString(int n) {
  string s;
  stringstream convert;
  convert << n;
  s = convert.str();
  return s;
} // NOTES:toString
// ENDTEMPLATE_BY_ZAYED

#define INF 2147483647
#define MOD 1000000007
const double PI = 2 * acos(0.0);
const double EPS = 1e-11;
const int SIZE = 1e6;

// fflush(stdout);
string S;

int main() {
  std::ios_base::sync_with_stdio(0);
  int tcases, I, J, K, N, n, m, cnt = 0, len;
  // READ();
  // WRITE();
  cin >> S;
  len = S.length();

  S[0] = toUpperCase(S[0]);
  cout << S << NL;

  return 0;
}
