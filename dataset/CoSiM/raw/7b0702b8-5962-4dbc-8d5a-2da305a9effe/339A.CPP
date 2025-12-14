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

// split a string to vector<string>
std::vector<std::string> split(const std::string &str, char delim) {
  std::istringstream stream(str);
  std::string element;
  std::vector<std::string> result;
  while (std::getline(stream, element, delim)) {
    result.push_back(element);
  }
  return result;
}

// join vector<string> to a string with delimiter
std::string join(const std::vector<std::string> &v,
                 const std::string delim = "") {
  std::string result;
  for (unsigned int i = 0; i < v.size(); ++i) {
    result += v[i];
    if (i != v.size() - 1) {
      result += delim;
    }
  }
  return result;
}

int main() {
  string in;
  cin >> in;
  vector<string> v = split(in, '+');
  sort(v.begin(), v.end());
  string answer = join(v, "+");
  cout << answer << endl;
  return 0;
}
