// 236A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int arr[27] = {0};
  int count = 0;
  for (int i = 0; i < s.length(); i++) {
    arr[s[i] - 97] = 1;
  }
  for (int i = 0; i < 26; i++) {
    if (arr[i] == 1)
      count++;
  }
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
