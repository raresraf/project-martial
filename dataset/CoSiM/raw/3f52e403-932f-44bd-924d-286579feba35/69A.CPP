#include "bits/stdc++.h"
#define do_not_sync                                                            \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define FIO                                                                    \
  freopen("input.txt", "r", stdin);                                            \
  freopen("output.txt", "w", stdout);

using namespace std;

int main(void) {
  do_not_sync
      // FIO
      int n;
  cin >> n;
  int sumx = 0, sumy = 0, sumz = 0;

  for (int i = 0; i < n; i++) {
    int x, y, z;
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }

  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
