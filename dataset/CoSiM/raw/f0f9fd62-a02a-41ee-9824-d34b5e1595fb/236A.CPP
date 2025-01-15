#include <iostream>
#include <string>

using namespace std;

int main() {
  string str;
  cin >> str;
  int a[26] = {0};
  int ctr = 0;
  for (int i = 0; i < str.length(); i++)
    a[str[i] - 'a']++;
  for (int i = 0; i < 26; i++) {
    if (a[i] < 1)
      ++ctr;
  }
  cout << (ctr % 2 == 0 ? "CHAT WITH HER!" : "IGNORE HIM!");
  return 0;
}