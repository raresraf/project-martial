#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n'
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  string s;
  cin >> s;
  for (int i = 1; i < s.length(); ++i) {
    if (s.find("1111111") != string::npos ||
        s.find("0000000") != string::npos) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO";
  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }