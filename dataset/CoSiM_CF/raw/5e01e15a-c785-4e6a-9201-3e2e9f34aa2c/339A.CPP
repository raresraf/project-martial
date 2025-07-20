#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  for (int i = 0; i < str.size(); i += 2) {
    for (int j = 0; j < i; j += 2) {
      if (str[i] < str[j]) {
        swap(str[i], str[j]);
      }
    }
  }
  cout << str;
}
