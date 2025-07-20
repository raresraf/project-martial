#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  string str;
  cin >> str;
  for (int i = 1; i <= t; i++) {
    for (int i = 1; i < str.length(); i++) {
      if (str[i] == 'G' && str[i - 1] == 'B') {
        swap(str[i], str[i - 1]);
        i++;
      }
    }
  }
  cout << str << endl;
}