#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  cin >> str;

  if (str.length() == 1)
    cout << str << endl;
  else {
    sort(str.begin(), str.end());
    int j = str.length() / 2;
    for (int i = j; i < str.length(); i++) {
      cout << str[i];
      if (i == str.length() - 1)
        break;
      else
        cout << "+";
    }
  }
}
