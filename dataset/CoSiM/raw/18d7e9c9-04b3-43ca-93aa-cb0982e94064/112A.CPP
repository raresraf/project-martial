#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {

  string s1;
  string s2;
  cin >> s1;
  cin >> s2;
  int i;
  int ans = 0;
  for (i = 0; i < s1.size(); i++) {

    if (tolower(s1[i]) < tolower(s2[i])) {
      ans = -1;
      break;
    } else if (tolower(s1[i]) > tolower(s2[i])) {
      ans = 1;
      break;
    }
  }

  cout << ans << endl;

  return 0;
}