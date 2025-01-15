
/*                     */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int x;
  string s1, s2;
  cin >> s1 >> s2;

  for (int i = 0; i < s1.size(); i++) {
    if (tolower(s1[i]) == tolower(s2[i]))
      x = 0;
    else if (tolower(s1[i]) < tolower(s2[i])) {
      x = -1;
      break;
    }

    else if (tolower(s1[i]) > tolower(s2[i])) {
      x = 1;
      break;
    }
  }
  cout << x << endl;

  return 0;
}
