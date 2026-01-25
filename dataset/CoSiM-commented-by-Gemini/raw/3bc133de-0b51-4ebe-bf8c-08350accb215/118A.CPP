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

bool isvowel(char c) {

  if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'A' ||
      c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'y' || c == 'Y') {

    return true;
  }

  return false;
}

int main() {

  cin.tie(0);
  ios::sync_with_stdio(false);
  Input();
  Solve();

  string s;
  cin >> s;

  string ans = "";

  int len = s.size();

  for (int i = 0; i < len; i++) {

    if (!isvowel(s[i])) {

      if (s[i] >= 97 && s[i] <= 122) {

        ans += '.';
        ans += s[i];
      }
      if (s[i] >= 65 && s[i] <= 90) {

        ans += '.';
        ans += s[i] + 32;
      }
    }
  }

  cout << ans << endl;

  return 0;
}
