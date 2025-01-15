#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     \
  int T;                                                                       \
  cin >> T;                                                                    \
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int
using namespace std;
int main() {
  int s;
  cin >> s;
  int ans = 0;
  while (s--) {
    int a, b, c, count = 0;
    cin >> a;
    if (a == 1)
      count++;
    cin >> b;
    if (b == 1)
      count++;
    cin >> c;
    if (c == 1)
      count++;
    if (count >= 2)
      ans++;
  }
  cout << ans;
}
