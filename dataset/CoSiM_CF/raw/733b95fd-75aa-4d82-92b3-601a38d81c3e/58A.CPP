#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  char HELLO[] = {'h', 'e', 'l', 'l', 'o'};
  cin >> str;
  int length = str.length();
  int var = 0;
  int dp = 0;

  for (int i = 0; i < length; ++i) {
    if (str[i] == HELLO[var]) {
      var++;
      dp++;
    }
    if (dp == 5) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO" << '\n';
  return 0;
}
