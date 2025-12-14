#include <bits/stdc++.h>

#define ll long long

using namespace std;

bool binary_search(ll low, ll high, ll key) {
  if (low <= high) {
    ll mid = low + (high - low) / 2;
    if (mid == key) {
      return true;
    } else if (mid > key) {
      return binary_search(low, mid - 1, key);
    } else {
      return binary_search(mid + 1, high, key);
    }
  } else {
    return false;
  }
}

bool isInteger(double a) { return floor(a) == a; }

int main() {
  ll n, a, b;
  cin >> n >> a >> b;
  bool flag = false;
  ll x, y;
  ll ans1, ans2;
  for (x = 0; x <= n / a; x++) {
    if (isInteger((n - a * x) * 1.0 / b)) {
      if (binary_search(0, n / b, (n - a * x) / b)) {
        flag = true;
        ans1 = x;
        ans2 = (n - a * x) / b;
        if (ans1 * a + ans2 * b == n) {
          break;
        } else {
          continue;
        }
      }
    }
  }

  if (flag) {
    cout << "YES" << endl;
    cout << ans1 << " " << ans2 << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}