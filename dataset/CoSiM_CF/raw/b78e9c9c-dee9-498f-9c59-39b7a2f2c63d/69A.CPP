#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n'
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  int n;
  cin >> n;

  int X = 0;
  int Y = 0;
  int Z = 0;
  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;
    X += x;
    Y += y;
    Z += z;
  }
  if (X == 0 && Y == 0 && Z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }