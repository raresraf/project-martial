#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  int n = str.length();
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j)
      if (str[j] < str[i] && str[j] != '+' && str[i] != '+')
        swap(str[i], str[j]);
  }
  cout << str;
  return 0;
}