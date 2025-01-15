/* samikshamodi - Samiksha Modi */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

#define pb push_back

#define sync                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
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
#define mem(arr, val) memset(arr, val, sizeof(arr));

const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const ll N = 1e5 + 7;

int main() {
  sync;
  int n, k;
  cin >> n >> k;
  int a[n];
  for (int i = 0; i < n; i += 1)
    cin >> a[i];

  int score = a[k - 1];
  int i = 0, cnt = 0;
  for (int i = 0; i < n; i++) {
    if (a[i] > 0 && a[i] >= score) {
      cnt++;
    } else
      break;
  }
  cout << cnt;
  return 0;
}