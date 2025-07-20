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
  string s;
  cin >> s;

  int arr[N];
  int iter = 0;
  FOR(i, 0, s.size(), 1) {
    if (isdigit(s[i]))
      arr[iter++] = s[i] - 48;
  }

  sort(arr, arr + iter);
  FOR(i, 0, iter, 1) {
    cout << arr[i];
    if (i != iter - 1)
      cout << "+";
  }
  // out1d(arr,iter);

  // cout<<arr[0]<<arr[1]<<arr[2];

  return 0;
}