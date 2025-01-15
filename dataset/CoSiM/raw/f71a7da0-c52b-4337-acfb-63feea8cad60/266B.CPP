#include <bits/stdc++.h>
using namespace std;
int i, j, n, k, t;
char q[100], temp[100];
int main()

{
  cin >> n >> t;
  cin >> q;
  while (t--) {
    for (i = 0; i < n; i++) {
      if (q[i] == 'B' && q[i + 1] == 'G') {
        q[i + 1] = 'B';
        q[i] = 'G';
        i++;
      }
    }
  }
  cout << q;
  return 0;
}
