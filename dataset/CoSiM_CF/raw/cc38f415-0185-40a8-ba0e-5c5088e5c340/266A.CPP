#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  int n;
  cin >> n;

  string s;
  cin >> s;

  int sum = 0;
  int len = 1;
  for (int i = 1; i < s.size(); ++i) {
    if (s[i - 1] == s[i])
      ++len;
    else {
      sum += len - 1;
      len = 1;
    }
  }

  sum += len - 1;

  cout << sum << endl;
}
