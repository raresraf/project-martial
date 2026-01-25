#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;

  cin >> s;

  int count_0 = 0, count_1 = 0, max_0 = 0, max_1 = 0;

  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '0') {
      count_1 = 0;
      count_0++;
      if (max_0 < count_0)
        max_0 = count_0;
      //	cout << max_0 << endl;
    } else {
      count_0 = 0;
      count_1++;
      if (max_1 < count_1)
        max_1 = count_1;
      //	cout << max_1 << endl;
    }
  }

  if ((max_0 >= 7) || (max_1 >= 7))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
