// Codeforces 266B - Queue at the School.cpp
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
#include <iomanip>
#include <numeric>
#include <utility>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

int caseno = 1;
#define NL '\n'
#define FOR(I, J, K) for (I = J; I < K; I++)
#define REV(I, J, K) for (I = J; I > K; I--)
#define SF scanf
#define PF printf
#define CLR(ar) memset(ar, 0, sizeof(ar))
#define SET(ar) memset(ar, -1, sizeof(ar))
#define PC() printf("Case %d: ", caseno++)
#define READ() freopen("in.txt", "r", stdin)
#define WRITE() freopen("out.txt", "w", stdout)
#define BOOST std::ios_base::sync_with_stdio(0);

typedef long long LL;              // NOTES:"%lld"
typedef unsigned long long ULL;    // NOTES:"%llu"
typedef long long int64;           // NOTES:int64
typedef unsigned long long uint64; // NOTES:uint64

#define INF 2147483647
#define MOD 1000000007
const double PI = 2 * acos(0.0);
const double EPS = 1e-11;
const int SIZE = 1e6;

string S;

int main() {
  BOOST
  int tcases, I, J, K, N, n, m, cnt = 0, len, t;
  /// READ();
  /// WRITE();
  cin >> len >> t;
  cin >> S;

  while (cnt < t) {
    for (I = 0; I < len - 1;) {
      if (S[I] == 'B' && S[I + 1] == 'G') {
        S[I + 1] = 'B';
        S[I] = 'G';
        I += 2;
      } else
        I++;
    }
    cnt++;
  }
  cout << S << NL;

  return 0;
}
