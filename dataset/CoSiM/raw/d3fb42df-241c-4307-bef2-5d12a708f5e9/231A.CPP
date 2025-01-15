#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n'
#define ll long long
#define PI 3.1415926535897932384626

void crap();
void solve();

int main() {
  crap();
  solve();
  return 0;
}

void solve() {
  int n;
  cin >> n;

  int count = 0;
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;
    if ((a + b + c) >= 2) {
      ++count;
    }
  }
  cout << count;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }