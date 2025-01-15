#include <cstring>
#include <iostream>

using namespace std;

int main() {
  string s;
  int h, e, l1, l2, o;
  int ans, i, j = 0;

  cin >> s;
  string test = "hello";

  int len = s.length();

  if (len < 5)
    ans = 0;
  else {
    for (i = 0; i < len; i++) {
      if ((test[j] == s[i])) {
        j++;
      }
    }
  }

  if (j == 5)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
