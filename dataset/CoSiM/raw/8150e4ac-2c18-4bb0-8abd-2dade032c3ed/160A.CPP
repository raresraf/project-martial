#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <utility>
#include <vector>

#define out freopen("output.txt", "w", stdout);
#define in freopen("input.txt", "r", stdin);
#define pub push_back
#define pob pop_back
#define infinity 2147483647
#define LL long long
#define Pi acos(-1)
#define SZ 1000000

/// int col[8] = {0, 1, 1, 1, 0, -1, -1, -1};
/// int row[8] = {1, 1, 0, -1, -1, -1, 0, 1};
/// int col[4] = {1, 0, -1, 0};
/// int row[4] = {0, 1, 0, -1};

using namespace std;

int main() {
  string s;
  int n, k, cnt = 0, ans = 0;
  scanf("%d %d", &n, &k);
  while (n--) {
    cin >> s;
    cnt = 0;
    for (int i = 0; i < s.size(); i++) {
      if (s[i] == '4' || s[i] == '7')
        cnt++;
    }
    if (cnt <= k)
      ans++;
  }
  printf("%d\n", ans);
  return 0;
}
