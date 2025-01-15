#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  int arr[26] = {0}, n(0);
  for (int i = 0; i < s.length(); i++) {
    arr[int(s[i]) - 97] = 1;
    // cout << "i" << int(s[i])-97 << endl;
  }
  for (int i = 0; i < 26; i++) {
    if (arr[i] == 1)
      n++;
  }
  // cout << n << endl;
  n % 2 == 0 ? cout << "CHAT WITH HER!" : cout << "IGNORE HIM!";
  return 0;
}
