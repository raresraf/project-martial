/* samikshamodi - Samiksha Modi */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

#define FOR(iterator, start, end, increment)                                   \
  for (int iterator = start; iterator < end; iterator += increment)
#define out1d(arr, size)                                                       \
  {                                                                            \
    for (int i = 0; i < size; i++)                                             \
      cout << arr[i] << " ";                                                   \
  }
#define in2d(arr, row, col)                                                    \
  for (int i = 0; i < row; i++)                                                \
    for (int j = 0; j < col; j++)                                              \
      cin >> arr[i][j];
#define out2d(arr, row, col)                                                   \
  {                                                                            \
    for (int i = 0; i < row; i++) {                                            \
      for (int j = 0; j < col; j++)                                            \
        cout << arr[i][j] << " ";                                              \
      cout << endl;                                                            \
    }                                                                          \
  }

const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const ll N = 1e5 + 7;

int main() {
  int n;
  cin >> n;
  int a[n];
  int x, tot = 0;
  FOR(i, 0, n, 1) {
    cin >> x;
    tot += x;
    a[i] = x;
  }
  sort(a, a + n);
  tot = tot / 2;
  int s = 0, cnt = 0;
  for (int i = n - 1; i >= 0; i--) {
    s = s + a[i];
    cnt++;
    if (s > tot)
      break;
  }
  cout << cnt;

  return 0;
}