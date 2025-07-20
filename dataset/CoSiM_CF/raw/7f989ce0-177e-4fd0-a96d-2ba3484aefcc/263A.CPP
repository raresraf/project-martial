#include <bits/stdc++.h>
using namespace std;
#define fo(i, n) for (i = 0; i < n; i++)
#define deb(x) cout << #x << "=" << x << endl

typedef vector<int> vi;
typedef vector<vi> vvi;
int main() {
  int i, j;
  vvi ar;
  int xx, y;
  int f = 1;
  fo(i, 5) {
    vi x(5);
    fo(j, 5) {
      cin >> x[j];
      if (x[j] == 1) {
        xx = i;
        y = j;
        f = 0;
        break;
      }
    }
    if (!f)
      break;
    ar.push_back(x);
  }

  xx++;
  y++;
  xx = abs(3 - xx);
  y = abs(3 - y);
  cout << xx + y;
  return 0;
}
