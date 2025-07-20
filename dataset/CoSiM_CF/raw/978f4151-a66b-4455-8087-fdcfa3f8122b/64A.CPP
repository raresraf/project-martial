#include <bits/stdc++.h>
#define IOS ios::sync_with_stdio(0);
#define pb push_back
#define mkp make_pair
#define all(a) a.begin(), a.end()
#define rall(a) a.rbegin(), a.rend()
#define maximum(a) *max_element(all(a))
#define minimum(a) *min_element(all(a))
#define MAX (int)2e5 + 7
using namespace std;
typedef long long LL;
typedef pair<int, int> pi;
typedef pair<LL, LL> pill;

int main() {
  IOS;
  int n, i, j;

  cin >> n;
  int a[n];
  bool flag = true;
  cin >> a[0];
  int cnt = 0;
  for (i = 1; i < n; i++) {
    cin >> a[i];
    if (a[i - 1] == 1) {
      if (a[i] == 2) {
        if (i > 2 && a[i - 2] == 3)
          cnt += 2;
        else
          cnt += 3;
      } else if (a[i] == 3)
        cnt += 4;
    } else if (a[i - 1] == 2) {
      if (a[i] == 1)
        cnt += 3;
      else
        flag = false;
    } else {
      if (a[i] == 1)
        cnt += 4;
      else
        flag = false;
    }
  }
  if (flag)
    cout << "Finite" << endl << cnt << endl;
  else
    cout << "Infinite" << endl;
}
