#include <bits/stdc++.h>
using namespace std;
#define F first
#define S second
#define mp make_pair
#define pb push_back
#define arsz (1 << 20)
typedef long long ll;
int main() {
  // freopen("input.txt","r", stdin);
  ios_base::sync_with_stdio(false);
  int ary[6][6];
  int x, y;
  for (int i = 1; i <= 5; i++)
    for (int j = 1; j <= 5; j++) {
      cin >> ary[i][j];
      if (ary[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  cout << abs(x - 3) + abs(y - 3) << endl;
}