#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n';
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int n, count;
  string s;
  cin >> n >> s;
  count = 0;
  for (int i = 1; i < n; ++i) {
    if (s[i] == s[i - 1]) {
      ++count;
    }
  }
  cout << count;

  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }