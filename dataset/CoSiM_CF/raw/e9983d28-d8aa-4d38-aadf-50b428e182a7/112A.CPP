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
// stoi-string to integer
// to_string(int)+"jhg"
// transform(sl.begin(), sl.end(), sl.begin(), ::tolower);
// set<int> s to remove duplicates
int main() {
  string s1, s2;
  cin >> s1 >> s2;
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  FOR(i, 0, s1.length(), 1) {
    if (s1[i] != s2[i]) {
      if (s1[i] < s2[i])
        cout << "-1";
      else
        cout << "1";
      break;

    } else if (s1[i] == s2[i] && i == s1.length() - 1)
      cout << "0";
  }
}