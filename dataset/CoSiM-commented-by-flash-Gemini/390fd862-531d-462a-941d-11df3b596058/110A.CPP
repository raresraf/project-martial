#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<int> vi;

string solve(ll n) {
  int cnt = 0;
  while (n > 0) {
    if (n % 10 == 4 || n % 10 == 7) {
      cnt++;
    }
    n /= 10;
  }

  return (cnt == 4 || cnt == 7) ? "YES" : "NO";
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  ll n;
  cin >> n;
  cout << solve(n) << endl;
}
