#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  int sum = 0, count[26] = {0};
  for (int i = 0; i < str.size(); i++) {
    count[str[i] - 97]++;
  }
  for (int i = 0; i < 26; i++) {
    if (count[i] > 0)
      ++sum;
  }
  if (sum % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
