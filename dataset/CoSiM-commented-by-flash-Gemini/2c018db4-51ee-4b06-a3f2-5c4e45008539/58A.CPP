#include <bits/stdc++.h>
using namespace std;
int main() {
  string str, word = "hello";
  int j = 0, count = 0;
  cin >> str;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == word[j]) {
      j++;
      count++;
    }
    if (count == 5)
      break;
  }
  if (count == 5)
    cout << "YES";
  else
    cout << "NO";
}
