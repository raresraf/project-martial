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
  string str;
  cin >> str;
  sort(str.begin(), str.end());
  int cnt = 1;
  FOR(i, 1, str.size(), 1) {
    if (str[i] != str[i - 1])
      cnt++;
  }
  if (cnt & 1 == 1) // odd
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";

  return 0;
}