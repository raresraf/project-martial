#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define REP(i, n) for (int i = 0; i < (int)(n); ++i)
#define FOR(i, m, n) for (int i = (m); i < (int)(n); ++i)
#define ALL(x) (x).begin(), (x).end()
#define INF 2000000000

#ifdef LOCAL
#define dprintf(...) fprintf(stdout, __VA_ARGS__)
#else
#define dprintf(...) 0
#endif

using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
typedef unsigned int uint;

const double PI = 3.1415926535897932384626433832795;

int main() {
  int n;
  cin >> n;
  vector<int> v(n);
  int total = 0;
  REP(i, n) {
    cin >> v[i];
    total += v[i];
  }
  sort(v.begin(), v.end(), greater<int>());
  int answer = 0;
  int current = 0;
  REP(i, v.size()) {
    answer++;
    current += v[i];
    if (current > total / 2) {
      break;
    }
  }
  cout << answer << endl;
  return 0;
}
