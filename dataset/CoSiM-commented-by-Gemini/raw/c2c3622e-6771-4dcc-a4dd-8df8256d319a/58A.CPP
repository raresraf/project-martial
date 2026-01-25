#include <algorithm>
#include <iostream>

using namespace std;

int main() {

  string s;
  cin >> s;
  int i, j;
  j = 0;
  string ans = "hello";
  for (i = 0; i < s.size(); i++) {

    if (s[i] == ans[j]) {

      if (j == 4) {
        cout << "YES" << endl;
        return 0;
      }
      j++;
    }
  }
  cout << "NO" << endl;

  return 0;
}
