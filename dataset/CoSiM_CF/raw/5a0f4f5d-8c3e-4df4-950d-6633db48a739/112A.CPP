// Link to Problem: https://codeforces.com/problemset/problem/112/A

#include <iostream>

using namespace std;

int main() {
  string s, t;
  cin >> s >> t;
  bool getout = false;
  for (int i = 0; i < s.length(); i++) {
    int si = s[i], ti = t[i];
    if (si < 97)
      si %= 65;
    else if (si > 96)
      si %= 97;

    if (ti < 97)
      ti %= 65;
    else if (ti > 96)
      ti %= 97;

    // cout<<si<<" "<<ti<<endl;

    if (si < ti) {
      cout << "-1";
      getout = true;
      break;
    } else if (si > ti) {
      cout << "1";
      getout = true;
      break;
    }
  }

  if (!getout)
    cout << "0";
  return 0;
}