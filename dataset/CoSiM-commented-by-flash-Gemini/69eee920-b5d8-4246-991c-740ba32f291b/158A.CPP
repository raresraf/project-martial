#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<ll> vi;
#define rep(i, a, b) for (int i = a; i < b; i++)

int main() {
  int n, k;
  cin >> n >> k;
  vi v(n);
  rep(i, 0, n) { cin >> v[i]; }

  int count = 0;
  rep(i, 0, n) {
    if (v[i] >= v[k - 1] && v[i] > 0) {
      count++;
    }
  }
  cout << count;
  return 0;
}
