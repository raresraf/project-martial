#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n';
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int n;
  cin >> n;

  while (n--) {
    string s;
    cin >> s;
    if (s.length() > 10) {
      cout << s.at(0) << s.length() - 2 << s.back() << '\n';
    } else {
      cout << s << '\n';
    }
  }
  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }