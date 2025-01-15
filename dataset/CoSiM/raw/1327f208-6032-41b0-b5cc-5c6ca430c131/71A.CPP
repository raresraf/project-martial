#include <bits/stdc++.h>

using namespace std;
#define EPS (1e-7)
#define INF (1e9)
#define PI (acos(-1))

typedef long long ll;
#define endl '\n'

int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};

const int MOD = 1000000007;

//////////////////////////////////////////

void Input() {}

void Solve() {}

int main() {

  cin.tie(0);
  ios::sync_with_stdio(false);
  // Input();
  // Solve();

  int t;

  cin >> t;

  while (t--) {

    string s;
    cin >> s;

    int n = s.size();
    if (n > 10) {

      cout << s[0] << n - 2 << s[n - 1] << endl;
    }

    else
      cout << s << endl;
  }

  return 0;
}
