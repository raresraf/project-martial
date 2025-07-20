#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  for (int tc = 0; tc < n; tc++) {
    string str;
    cin >> str;

    int strlen = str.length();

    if (strlen > 10) {
      std::string mid = to_string((strlen - 2));
      string abbv = str[0] + mid + str[strlen - 1];
      cout << abbv;

    } else {
      cout << str;
    }
    cout << endl;
  }

  return 0;
}
