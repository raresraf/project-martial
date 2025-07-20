#include <bits/stdc++.h>
using namespace std;

#define FOR(iterator, start, end, increment)                                   \
  for (int iterator = start; iterator < end; iterator += increment)
#define out2d(arr, row, col)                                                   \
  {                                                                            \
    for (int i = 0; i < row; i++) {                                            \
      for (int j = 0; j < col; j++)                                            \
        cout << arr[i][j] << " ";                                              \
      cout << endl;                                                            \
    }                                                                          \
  }
#define in2d(arr, row, col)                                                    \
  for (int i = 0; i < row; i++)                                                \
    for (int j = 0; j < col; j++)                                              \
      cin >> arr[i][j];

int main() {
  int a[5][5];
  in2d(a, 5, 5);
  int ii, jj;

  FOR(i, 0, 5, 1) {
    FOR(j, 0, 5, 1) {
      if (a[i][j] == 1) {
        ii = i;
        jj = j;
      }
    }
  }

  cout << abs(ii - 2) + abs(jj - 2);

  return 0;
}