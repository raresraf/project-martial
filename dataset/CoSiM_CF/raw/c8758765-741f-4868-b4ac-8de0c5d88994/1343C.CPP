#include <algorithm>
#include <bitset>
#include <climits>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ull unsigned long long
#define ll long long

using namespace std;

int main() {
  int t;
  cin >> t;
  while (t--) {
    int n;
    cin >> n;
    ll ans = 0;
    int a;
    cin >> a;
    bool pos = a > 0;
    int compare = a;
    n--;
    while (n--) {
      cin >> a;
      if (pos && a > 0 || !pos && a < 0) {
        compare = max(compare, a);
      } else {
        ans += compare;
        compare = a;
        pos = a > 0;
      }
    }

    ans += compare;
    cout << ans << endl;
  }
  return 0;
}