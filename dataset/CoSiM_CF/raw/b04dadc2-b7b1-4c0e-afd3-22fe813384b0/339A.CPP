#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int counts[4]{0, 0, 0, 0};
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i += 2) {
    ++counts[s[i] - '1'];
  }

  for (int i = 0; i < 3;) {
    while (i <= 3 && counts[i] == 0)
      ++i;
    while (counts[i] > 0) {
      cout << (1 + i);
      if (counts[i] > 1)
        cout << "+";
      --counts[i];
    }
    while (i < 3 && counts[i] == 0)
      ++i;
    if (i < 3 && counts[i] > 0)
      cout << "+";
  }
  cout << endl;
  return 0;
}