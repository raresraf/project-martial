#include <iostream>
using namespace std;

#define FOR(iterator, start, end, increment)                                   \
  for (int iterator = start; iterator < end; iterator += increment)

int main() {
#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  int n, a[1001][3], cnt, que = 0;

  cin >> n;

  FOR(i, 0, n, 1) {
    FOR(j, 0, 3, 1)
    cin >> a[i][j];
  }

  FOR(i, 0, n, 1) {
    cnt = 0;
    FOR(j, 0, 3, 1) {
      if (a[i][j] == 1)
        cnt++;
      if (cnt == 2) {
        que++;
        break;
      }
    }
  }

  cout << que;

  return 0;
}