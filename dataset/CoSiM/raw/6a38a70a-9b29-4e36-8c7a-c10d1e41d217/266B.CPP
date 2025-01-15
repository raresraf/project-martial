#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  string str;
  cin >> n >> t;
  cin >> str;
  for (int i = 1; i <= t; i++) {
    for (int j = 1; j < str.size(); j++) {
      if (str[j] == 'G' && str[j - 1] == 'B') {
        swap(str[j], str[j - 1]);
        j++;
      }
    }
  }
  cout << str << endl;
  return 0;
}
