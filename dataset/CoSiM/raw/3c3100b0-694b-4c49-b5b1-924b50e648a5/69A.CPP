#include <bits/stdc++.h>
using namespace std;
#define F first
#define S second
#define mp make_pair
#define pb push_back
typedef long long ll;

int main() {
  // freopen("input.txt","r", stdin);
  ios_base::sync_with_stdio(false);
  int n;
  cin >> n;
  int xs = 0, ys = 0, zs = 0;
  for (int i = 0; i < n; i++) {
    int x, y, z;
    cin >> x >> y >> z;
    xs += x, ys += y, zs += z;
  }
  if (xs == 0 && ys == 0 && zs == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}