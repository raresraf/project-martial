#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n'
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  ll n;
  int count = 0;

  cin >> n;
  string nString = to_string(n);
  for (int i = 0; i < nString.length(); ++i) {
    if (nString.at(i) == '4' || nString.at(i) == '7') {
      ++count;
    }
  }
  if (count == 4 || count == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }