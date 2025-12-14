#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  int flag = 0, i = 0;

  cin >> str;
  while (i <= str.size()) {
    if (str[i] == str[i - 1]) {
      flag++;
      if (flag == 6) {
        cout << "YES" << endl;
        return 0;
      }
    }

    else
      flag = 0;

    i++;
  }
  cout << "NO" << endl;

  return 0;
}
