#include <bits/stdc++.h>
using namespace std;
int main() {
  int query;
  cin >> query;
  int num = 0;
  while (query--) {
    string s;
    cin >> s;
    int len = s.size();
    int i;
    char at = '+';
    char am = '-';
    int v = 0, g = 0;
    for (i = 0; i < len; i++) {
      if (s[i] == at) {
        v++;
      } else if (s[i] == am) {
        g++;
      }
    }
    if (v == 2) {
      num++;
    } else if (g == 2) {
      num--;
    }
    //   cout<<num<<endl;
  }
  cout << num << endl;
}