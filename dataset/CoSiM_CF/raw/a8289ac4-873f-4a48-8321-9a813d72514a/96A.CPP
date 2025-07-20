#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;
  int one = 0, zero = 0;
  int ans = 0;
  if (s[0] == '1')
    one++;
  else
    zero++;
  for (int i = 1; i < s.size(); i++) {
    if (s[i] == s[i - 1])
      if (s[i] == '1')
        one++;
      else
        zero++;
    else {
      if (s[i] == '1') {
        if (zero > ans)
          ans = zero;
        one++;
        zero = 0;
      } else {
        if (one > ans)
          ans = one;
        zero++;
        one = 0;
      }
    }
  }
  if (max(zero, one) > ans)
    ans = max(zero, one);
  if (ans >= 7)
    cout << "YES";
  else
    cout << "NO";
}