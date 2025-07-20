#include <bits/stdc++.h>
using namespace std;

void swap(char &a, char &b) {
  char temp;
  temp = a;
  a = b;
  b = temp;
}
int main() {
  int s, t, f = 0;
  cin >> s >> t;
  string str;
  cin >> str;
  for (int i = 0; i < t; ++i) {
    for (int j = 0; j < s; ++j) {
      if (str[j] == 'B' && str[j + 1] == 'G') {
        swap(str[j], str[j + 1]);
        j++;
      }
    }
  }

  cout << str;
}