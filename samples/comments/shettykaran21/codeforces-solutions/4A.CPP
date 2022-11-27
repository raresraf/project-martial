#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n';
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int w;
  cin >> w;
  if (w % 2 == 0 && w > 2) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }