#include <algorithm>
#include <bitset>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>
using namespace std;

#define sci(x) scanf("%d", &x)
#define scli(x) scanf("%lld", &x)
#define sc2i(x, y) scanf("%d%d", &x, &y)
#define sc2li(x, y) scanf("%lld%lld", &x, &y)
#define pb push_back
#define X first
#define Y second
#define boost                                                                  \
  cin.tie(0);                                                                  \
  cout.tie(0)
#define REP(i, a, b) for (int i = a; i < b; i++)
#define pf printf
#define MEM(a, val) memset(a, val, sizeof(a))
#define pi 3.14159265359
#define mp make_pair
#define all(c) c.begin(), c.end() // eg sort(all(v));

typedef long long ll;
typedef pair<int, int> pii;
typedef unsigned long long ull;
typedef vector<int> vi;

int main() {
  int n, t;
  string s;
  cin >> n >> t;
  cin >> s;
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1;) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j] = 'G';
        s[j + 1] = 'B';
        j += 2;
      } else
        j++;
    }
  }
  cout << s << endl;
  return 0;
}
