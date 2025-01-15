#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

#define endl "\n"
#define si(a) scanf("%d", &a)
#define sii(a, b) scanf("%d %d", &a, &b)
#define siii(a, b, c) scanf("%d %d %d", &a, &b, &c)
#define sl(a) scanf("%lld", &a)
#define sll(a, b) scanf("%lld %lld", &a, &b)
#define pf printf
#define vi vector<int>
#define vii vector<pair<int, int>>
#define pii pair<int, int>
#define all(c) c.begin(), c.end()
#define ms(a, c) memset(a, c, sizeof(a))
#define tr(c, i)                                                               \
  for (decltype((c).begin()) i = (c).begin(); i != (c).end(); i++)
#define trr(c, i)                                                              \
  for (decltype((c).rbegin()) i = (c).rbegin(); i != (c).rend(); i++)

using namespace std;

#define N 1001
#define mod 1000000007
#define inf 1e9

typedef long long ll;
typedef unsigned long long ull;

int power(int a, int b) {
  int res = 1;
  while (b) {
    if (b & 1)
      res *= a;
    a *= a;
    b >>= 1;
  }
  return res;
}

int dp[N], mp[N];
bool flag[N];

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  //    freopen("Input.txt","r",stdin);
  //    freopen("Output.txt","w",stdout);
  flag[0] = true;
  for (int i = 1; i < N; i++) {
    for (int j = N - 1; j - i >= 0; j--) {
      if (flag[j - i] && dp[j] < dp[j - i] + 1)
        dp[j] = dp[j - i] + 1, mp[j] = j - i, flag[j] = true;
    }
  }
  int n;
  cin >> n;
  cout << dp[n] << endl;
  while (n)
    cout << n - mp[n] << ' ', n = mp[n];
  cout << endl;
  return 0;
}
