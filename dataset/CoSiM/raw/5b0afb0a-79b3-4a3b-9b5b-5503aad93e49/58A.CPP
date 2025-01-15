// 56636198	Practice:havoccboi	58A - 10	GNU C++17	Accepted
// 30 ms	8 KB	2019-07-06 22:43:18	2019-07-06 22:43:18

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s = "hello";
  string in;
  cin >> in;

  int i = 0;
  for (char ch : in) {
    if (i == 5)
      break;
    if (ch == s[i]) {
      i++;
    }
  }
  if (i == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
