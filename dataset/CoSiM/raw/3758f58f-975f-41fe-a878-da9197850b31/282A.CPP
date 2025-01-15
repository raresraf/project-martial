#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n';
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int n, x;
  x = 0;
  cin >> n;

  while (n--) {
    string s;
    cin >> s;
    if (s[1] == '+') {
      ++x;
    } else if (s[1] == '-') {
      --x;
    }
  }
  cout << x;

  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }