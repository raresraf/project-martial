#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// using
using namespace std;

// typedef
typedef vector<int> VI;
typedef vector<VI> VII;
typedef vector<string> VS;
typedef vector<long> VL;
typedef long long LL;
typedef vector<LL> VLL;

// container utils
#define PB push_back
#define PF push_front
#define GRT(x) greater<(x)>()
#define ASORT(x) sort((x).begin(), (x).end())
#define DSORT(x, y) sort((x).begin(), (x).end(), greater<(y)>())
#define FILL(x, y) fill((x).begin(), (x).end(), (y))
#define COPY(x, y)                                                             \
  (y).clear();                                                                 \
  copy((x).begin(), (x).end(), back_inserter(y))

// repetition
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define FORL(i, a, b) for (long i = (a); i < (b); i++)
#define REP(i, n) FOR(i, 0, n)
#define REPL(i, n) FORL(i, 0, n)

// debug
#define DUMP(a) cerr << #a << " = " << (a) << endl
#define DUMP2(a, b)                                                            \
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << endl
#define DUMP3(a, b, c)                                                         \
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << ", " << #c       \
       << " = " << (c) << endl

bool isVowel(char c) {
  switch (c) {
  case 'a':
  case 'e':
  case 'i':
  case 'o':
  case 'u':
  case 'y':
    return true;
    break;
  default:
    return false;
    break;
  }
}

string rmVowels(string str) {
  string rtn = "";

  REP(i, str.size()) {
    if (!isVowel(str[i]))
      rtn += str[i];
  }

  return rtn;
}

string toLower(string str) {
  string rtn = "";

  REP(i, str.size()) {
    if (str[i] >= 'A' && str[i] <= 'Z') {
      rtn += (str[i] + ('a' - 'A'));
    } else {
      rtn += str[i];
    }
  }

  return rtn;
}

string insertDots(string str) {
  string rtn = "";

  REP(i, str.size()) {
    rtn += ".";
    rtn += str[i];
  }

  return rtn;
}

void solve() {
  string str;
  cin >> str;

  str = toLower(str);

  str = rmVowels(str);

  str = insertDots(str);

  cout << str << endl;
}

int main(int argc, char *argv[]) {
  solve();

  return 0;
}
