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

// whether x is in v or not
template <class T> bool inVector(const T &x, const std::vector<T> &v) {
  auto iter = std::find(v.begin(), v.end(), x);
  if (iter != v.end()) {
    return true;
  } else {
    return false;
  }
}

int main() {
  vector<char> vowels = {'a', 'o', 'y', 'e', 'u', 'i'};
  string input;
  cin >> input;
  transform(input.begin(), input.end(), input.begin(), ::tolower);

  string result;
  REP(i, input.size()) {
    if (!inVector(input[i], vowels)) {
      result.push_back('.');
      result.push_back(input[i]);
    }
  }

  cout << result << endl;

  return 0;
}
