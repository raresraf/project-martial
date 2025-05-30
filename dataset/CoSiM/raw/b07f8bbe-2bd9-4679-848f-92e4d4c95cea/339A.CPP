/*
        SubmissionId	:	4353166
        ContestId	:	339
        Index	:	A
        ProblemName	:	Helpful Maths
        ProblemTags	:	['greedy', 'implementation', 'sortings',
   'strings'] ProgrammingLanguage	:	GNU C++ Verdict	:	OK
*/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

using namespace std;

typedef long long ll;
typedef pair<int, int> pii;
#define X first
#define Y second
#define pb push_back
#define mp make_pair
#define SZ(x) ((int)((x).size()))
#define ALL(x) (x).begin(), (x).end()
#define debug(x) cerr << #x << " = " << x << endl
#define FOREACH(i, c)                                                          \
  for (__typeof((c).begin()) i = (c).begin(); i != (c).end(); i++)
#define FOR(i, a, n) for (__typeof(n) i = (a); i <= (n); i++)
#define FORD(i, n, a) for (__typeof(n) i = (n); i >= (a); i--)
#define MAXN 100010

vector<int> v;
char c;

int main() {
  ios::sync_with_stdio(false);
  while (cin >> c) {
    if (c != '+')
      v.pb(c - 48);
  }
  sort(ALL(v));
  cout << v[0];
  FOR(i, 1, SZ(v) - 1)
  cout << "+" << v[i];
  return 0;
}
