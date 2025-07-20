#include <algorithm>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <stack>
#include <stdlib.h>
#include <string.h>
#include <vector>
#define pb push_back
#define ins insert
#define mp make_pair
#define ll long long
#define max3(a, b, c) max(c, max(a, b))
#define min3(a, b, c) min(c, min(a, b))
#define M 998244353
using namespace std;
ll power[200005] = {1};
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  ll n, x = 0;
  cin >> n;
  for (int i = 1; i < 200005; ++i) {
    power[i] = (10 * power[i - 1]) % M;
  }
  for (int i = 1; i <= n - 2; ++i) {
    x = 81 * (n - i - 1);
    x = (x * power[n - i - 1]) % M;
    x = (x + 2 * power[n - i] * 9) % M;
    cout << x % M << " ";
  }
  if (n >= 2)
    cout << 180 << " ";
  cout << 10;

  return 0;
}