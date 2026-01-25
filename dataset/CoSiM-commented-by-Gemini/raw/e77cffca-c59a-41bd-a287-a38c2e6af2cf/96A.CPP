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

string s;
void Input() { cin >> s; }

void Solve() {

  char c = s[0];
  int cnt = 1;
  int len = s.size();
  for (int i = 1; i < len; i++) {

    if (cnt >= 7) {

      break;
    }
    if (c == s[i]) {

      cnt++;

    } else {

      c = s[i];
      cnt = 1;
    }
  }

  if (cnt >= 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}

int main() {

  cin.tie(0);
  ios::sync_with_stdio(false);
  Input();
  Solve();

  return 0;
}
